"""CPU-only check of TF_GLM_EFFORT_TAIL (patch 0096, issue #93): the reasoning-effort line rendered at the tail.

  python3 -B tools/test_effort_tail.py --source-root /path/to/patched/src [--model-dir DIR] [--expect-stock]
The checkpoint's own chat_template.jinja is read (read-only) from --model-dir, or from the Hugging Face cache
($HF_CACHE, $HF_HOME or ~/.cache/huggingface) snapshot of Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold. Needs jinja2 only;
token prefixes are compared too when `tokenizers` is installed. --expect-stock, on the unpatched source, shows the
reported behaviour: with the flag set, the conversation still differs from token 3 on between effort settings.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

MODEL = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
OPENED = "<|assistant|><think>"
SETTINGS = {"max": (True, "max"), "default": (True, None), "high": (True, "high"), "low": (True, "low"),
            "off": (False, None)}
LINE = {"max": "Max", "default": "Max", "high": "High", "low": "Low"}

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
CONVERSATIONS = {
    "chat": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "What does main.py do?"},
        {"role": "assistant", "content": "It starts the server.", "reasoning_content": "Let me think about main.py."},
        {"role": "user", "content": "And utils.py?"}],
    "tools": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Read a.txt and summarise it."},
        {"role": "assistant", "content": "", "reasoning_content": "I need the file first.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello world"},
        {"role": "assistant", "content": "It says hello world."},
        {"role": "user", "content": "Now read b.txt."}],
}


def model_dir(arg):
    if arg:
        return Path(arg)
    root = Path(os.environ.get("HF_CACHE") or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    snaps = sorted((root / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots").glob("*/chat_template.jinja"))
    if not snaps:
        sys.exit(f"no chat_template.jinja for {MODEL} under {root}: pass --model-dir")
    return snaps[-1].parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.families.glm5_next import prompts
    from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate

    mdir = model_dir(args.model_dir)
    inner = ChatTemplate(mdir)
    wrapped = ThinkingOffTemplate(inner)
    print(f"template: {mdir / 'chat_template.jinja'}")

    def render(messages, name, tools, flag):
        think, effort = SETTINGS[name]
        if flag is None:
            os.environ.pop("TF_GLM_EFFORT_TAIL", None)
        else:
            os.environ["TF_GLM_EFFORT_TAIL"] = flag
        extra = {"reasoning_effort": effort} if effort else None
        return wrapped.render(copy.deepcopy(messages), tools=tools, enable_thinking=think, extra=extra)

    def reference(messages, name, tools):
        """The text before the patch: the checkpoint template, then thinking_off's line removal and empty think block."""
        think, effort = SETTINGS[name]
        text = inner.render(copy.deepcopy(messages), tools=tools, enable_thinking=think,
                            extra={"clear_thinking": False, **({"reasoning_effort": effort} if effort else {})})
        if think:
            return text
        text = text.replace("<|system|>Reasoning Effort: Max", "", 1)
        return text + "</think>" if text.endswith(OPENED) else text

    def before_last_user(text):
        return text[:text.rindex("<|user|>")]

    tok = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    except Exception:  # noqa: BLE001 - tokenizers is optional here
        print("note: `tokenizers` not available, comparing text prefixes only")

    for cname, messages in CONVERSATIONS.items():
        tools = TOOLS if cname == "tools" else None
        # (a) flag unset and 0: byte-identical to the template as it rendered before, line at token 3
        for flag in (None, "0"):
            for name in SETTINGS:
                got = render(messages, name, tools, flag)
                assert got == reference(messages, name, tools), (cname, name, flag, "differs from the stock render")
                if SETTINGS[name][0]:
                    assert got.startswith(f"[gMASK]<sop><|system|>Reasoning Effort: {LINE[name]}"), (cname, name)
        if args.expect_stock:
            tails = {n: before_last_user(render(messages, n, tools, "1")) for n in SETTINGS}
            assert len(set(tails.values())) > 1, "stock source already keeps the prefix identical"
            print(f"{cname}: stock, flag set: the prefix still differs between settings, as reported")
            continue
        # (b) flag on: everything before the generation prompt is the same for every effort and thinking setting
        outs = {n: render(messages, n, tools, "1") for n in SETTINGS}
        prefixes = {n: before_last_user(t) for n, t in outs.items()}
        assert len(set(prefixes.values())) == 1, (cname, "prefix up to the last user turn differs",
                                                  {n: p[:60] for n, p in prefixes.items()})
        for n, t in outs.items():
            think = SETTINGS[n][0]
            if think:
                assert t.count("Reasoning Effort:") == 1, (cname, n, "effort line count", t.count("Reasoning Effort:"))
                assert t.endswith(f"<|system|>Reasoning Effort: {LINE[n]}{OPENED}"), (cname, n, t[-80:])
                assert not t.startswith("[gMASK]<sop><|system|>Reasoning Effort"), (cname, n, "still at the head")
                # same prompt as before with the line moved: put it back at the head and it is the stock text
                moved = t[:-len(f"<|system|>Reasoning Effort: {LINE[n]}{OPENED}")] + OPENED
                assert "[gMASK]<sop><|system|>Reasoning Effort: " + LINE[n] + moved[len("[gMASK]<sop>"):] == \
                    reference(messages, n, tools), (cname, n, "moving the line back does not give the stock text")
            else:
                assert t.count("Reasoning Effort:") == 0, (cname, n, "off keeps an effort line")
                assert t == reference(messages, n, tools), (cname, n, "off differs from the stock render")
                assert t.endswith(OPENED + "</think>"), (cname, n)
        assert outs["max"] == outs["default"], (cname, "no effort is Max")
        assert len({outs[n] for n in ("max", "high", "low")}) == 3, (cname, "efforts render alike")
        if tok is not None:
            ids = {n: tok.encode(t, add_special_tokens=False).ids for n, t in outs.items()}
            cut = len(tok.encode(prefixes["max"], add_special_tokens=False).ids)
            for n, i in ids.items():
                assert i[:cut] == ids["max"][:cut], (cname, n, "token prefix differs")
            common = min(len(i) for i in ids.values())
            first = {n: next((k for k in range(common) if i[k] != ids["max"][k]), common) for n, i in ids.items()}
            print(f"{cname}: first token that differs from max: {first} of {len(ids['max'])}")
        print(f"{cname}: flag off identical to stock for 5 settings; flag on: {len(prefixes['max'])} chars before the "
              f"last user turn identical across {len(SETTINGS)} settings, the effort line once, at the tail")

    if args.expect_stock:
        return

    # the tokenizer the Mac server and /tokenize-style callers use renders the same
    class Inner:
        def apply_chat_template(self, messages, tokenize=False, **kw):
            think = kw.get("enable_thinking", True)
            extra = {"clear_thinking": False, **({"reasoning_effort": kw["reasoning_effort"]} if "reasoning_effort" in kw else {})}
            text = inner.render(messages, tools=kw.get("tools"), enable_thinking=think, extra=extra)
            return self.encode(text) if tokenize else text

        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    gt = prompts.GlmTokenizer(Inner())
    messages = CONVERSATIONS["tools"]
    for flag in ("0", "1"):
        os.environ["TF_GLM_EFFORT_TAIL"] = flag
        for name, (think, effort) in SETTINGS.items():
            kw = {"enable_thinking": think, "add_generation_prompt": True, "tools": TOOLS}
            if effort:
                kw["reasoning_effort"] = effort
            got = gt.apply_chat_template(copy.deepcopy(messages), tokenize=False, **kw)
            want = render(messages, name, TOOLS, flag)
            assert got == want, (flag, name, "GlmTokenizer and the CUDA template disagree")
            assert gt.apply_chat_template(copy.deepcopy(messages), tokenize=True, **kw) == list(want.encode())
    os.environ["TF_GLM_EFFORT_TAIL"] = "yes"
    try:
        prompts.effort_tail_enabled()
        raise AssertionError("TF_GLM_EFFORT_TAIL=yes was accepted")
    except ValueError:
        pass
    # an effort line that is not at the head (client text quoting it) is left alone
    os.environ["TF_GLM_EFFORT_TAIL"] = "1"
    quoted = "[gMASK]<sop><|user|>say <|system|>Reasoning Effort: Low<|assistant|><think>"
    assert prompts.effort_to_tail(quoted, True) == quoted
    print("GlmTokenizer agrees with the CUDA template; invalid values refused; quoted lines untouched")
    print("effort tail: OK")


if __name__ == "__main__":
    main()
