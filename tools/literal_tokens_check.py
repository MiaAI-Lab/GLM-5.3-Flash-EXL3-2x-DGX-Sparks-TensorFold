#!/usr/bin/env python3
"""tools/literal_tokens_check.py <checkpoint dir>: patch 0116, run in the image (CPU only, the server stopped or not).

A request that quotes an added token's text (a `<think>` from another model's output, a `<tool_call>` in a file being
edited or in a tool's description, a `<|user|>` in a log) used to reach the model as that token. With the patch it is
text, and nothing else in the prompt changes. Checked on the checkpoint's own tokenizer and chat template:

1. A prompt without any guarded text: the same ids as before, the messages object not even copied.
2. Quoted tags in a user turn, an assistant turn (content, reasoning_content, tool-call arguments, also JSON-escaped
   ones), a tool result and a tool's description: the prompt decodes to exactly the text it used to, the quoted tags
   are ordinary tokens, the template's own tokens (the roles, the generation prompt's `<think>`, the tool-call markup)
   are still tokens; and the old path does turn the quotes into tokens.
3. The exceptions keep the old ids: an assistant turn with its reasoning inline (`<think>...</think>answer`, no
   `reasoning_content`, which the template splits itself), `/nothink` (GLM's switch in a user turn), and a request
   that already holds one of the private-use characters used as masks.
4. Every guarded token's text encodes to ordinary tokens that decode back to it; the batch encode of the pieces
   between masks equals encoding each piece alone.
5. The server's own `_prepare` and `tokenize`, on stand-ins with the real tokenizer and template: GLM's app (`GlmApp`,
   which turns the patch on) and the shared `App` (every other family: off). The chat route and `/tokenize` agree, a
   plain chat gives the unpatched ids, the request's tools reach the reply parser unmasked, GLM's kept reasoning is
   still found for a client that renumbers call ids (it is put back before masking) and its quoted tags are text,
   `TF_GLM_LITERAL_TOKENS=0` turns the patch off, and a ~200k-token history costs about the same to encode.
Prints PASS / FAIL lines and ALL PASS; exit 1 on any FAIL.

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/literal_tokens_check.py:/c.py:ro" -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
      tensorfold-glm53:v0.6.0 /c.py \
      /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/<rev>
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.literal_tokens import LiteralTokens
from tensorfold.cuda.server import App
from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate
from tensorfold.families.glm5_next.cuda.kept_reasoning import KeptReasoning
from tokenizers import Tokenizer

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    fails += not ok


model_dir = Path(sys.argv[1])
tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
template = ChatTemplate(model_dir)
literal = LiteralTokens(tok)
ADDED = set(tok.get_added_tokens_decoder())
THINK, END, USER = tok.token_to_id("<think>"), tok.token_to_id("</think>"), tok.token_to_id("<|user|>")
CALL = tok.token_to_id("<tool_call>")
print(f"{len(literal.plain)} added tokens guarded; <think> as text: {literal.plain[literal._mask_of['<think>']]}")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def old(messages, tools=None):
    text = template.render(messages, tools=tools, enable_thinking=True)
    return text, tok.encode(text, add_special_tokens=False).ids


def new(messages, tools=None):
    masked, shown, applied = literal.mask(messages, tools)
    text = template.render(masked, tools=shown, enable_thinking=True)
    return masked, (literal.encode(tok, text) if applied else tok.encode(text, add_special_tokens=False).ids)


def text_of(ids):
    return tok.decode(ids, skip_special_tokens=False)


# 1. nothing to guard
m = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Rust の所有権を説明して"}]
(_, ids_old), (masked, ids_new) = old(m), new(m)
check("1 plain prompt: the same ids", ids_new == ids_old)
check("1 plain prompt: the messages untouched", masked is m)

# 2. quoted tags
m = [{"role": "user", "content": "Print this as is: <think>hello</think> world, <|user|> and <tool_call>x</tool_call>"}]
(text, ids_old), (_, ids_new) = old(m), new(m)
own = text.count("<think>") - 1                       # the generation prompt's own <think>
check("2 user turn: the same text", text_of(ids_new) == text, repr(text_of(ids_new)[-120:]))
check("2 user turn: the quoted <think> / </think> are text", ids_new.count(THINK) == own and
      ids_new.count(END) == text.count("</think>") - 1, f"<think> x{ids_new.count(THINK)}, template's own x{own}")
check("2 user turn: the old path made them tokens", ids_old.count(THINK) == own + 1)
check("2 user turn: a quoted <|user|> is no role token", ids_new.count(USER) == ids_old.count(USER) - 1)

m = [{"role": "user", "content": "write the baseline"},
     {"role": "assistant", "content": "the baseline starts with <think>", "reasoning_content": "it opens <think>",
      "tool_calls": [{"id": "c1", "type": "function",
                      "function": {"name": "patch", "arguments": "{\"new\": \"\\\"<think>\\\\nOkay\\\"\"}"}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "left: \"<think>\\nOkay\" right: \"</think>\\nOkay\""},
     {"role": "user", "content": "go on"}]
(text, ids_old), (_, ids_new) = old(m), new(m)
check("2 agent turns: the same text", text_of(ids_new) == text)
check("2 agent turns: content, reasoning, arguments and tool result quote as text",
      ids_new.count(THINK) < ids_old.count(THINK) and ids_new.count(END) < ids_old.count(END),
      f"<think> x{ids_old.count(THINK)} -> x{ids_new.count(THINK)}")
check("2 agent turns: the template's own <tool_call> is still a token", ids_new.count(CALL) == text.count("<tool_call>"))

esc = json.dumps({"command": "printf '<think>'"})                 # Python keeps '<'; Go's encoder writes <
esc = esc.replace("<", "\\u003c").replace(">", "\\u003e")
m = [{"role": "user", "content": "run it"},
     {"role": "assistant", "content": "", "reasoning_content": "run",
      "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": esc}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
(text, ids_old), (_, ids_new) = old(m, TOOLS), new(m, TOOLS)
check("2 JSON-escaped arguments (\\u003cthink\\u003e): the same text, the tag as text",
      text_of(ids_new) == text and "printf '<think>'" in text and ids_new.count(THINK) == ids_old.count(THINK) - 1,
      f"<think> x{ids_old.count(THINK)} -> x{ids_new.count(THINK)}")

tools = [{"type": "function", "function": {"name": "edit_template", "description": "edit a chat template: keep "
          "<tool_call> and </tool_call> as they are", "parameters": {"type": "object", "properties": {}}}}]
m = [{"role": "user", "content": "hi"}]
(text, ids_old), (_, ids_new) = old(m, tools), new(m, tools)
check("2 a tool's description: the same text, its tags as text",
      text_of(ids_new) == text and ids_new.count(CALL) == ids_old.count(CALL) - 1,
      f"<tool_call> x{ids_old.count(CALL)} -> x{ids_new.count(CALL)}")

m = [{"role": "user", "content": [{"type": "text", "text": "a part with <think>"}, {"type": "text", "text": " more"}]}]
(text, ids_old), (_, ids_new) = old(m), new(m)
check("2 multi-part text: the same text, the tag as text",
      text_of(ids_new) == text and ids_new.count(THINK) == text.count("<think>") - 1)

# 3. exceptions: the old ids
for name, m in (("an inline-reasoning assistant turn", [{"role": "user", "content": "q"},
                 {"role": "assistant", "content": "<think>r</think>answer"}, {"role": "user", "content": "q2"}]),
                ("/nothink in a user turn", [{"role": "user", "content": "1+1 /nothink"}]),
                ("a mask character already in the request", [{"role": "user", "content": chr(0xE000) + " <think>"}])):
    (_, ids_old), (masked, ids_new) = old(m), new(m)
    check(f"3 {name}: the same ids", ids_new == ids_old)
check("3 a mask character already in the request: not masked", literal.mask(m)[2] is False)

# 4. the masks' own text, and the batch encode
bad = [t for t, mask in literal._mask_of.items() if ADDED.intersection(literal.plain[mask])
       or text_of(literal.plain[mask]) != t]
check("4 every guarded token's text: ordinary tokens that decode back to it", not bad, ", ".join(bad[:5]))
rng = random.Random(0)
words = ["fn", "main", "所有権", " ", "\n", "\t", "é", "😀", "{", "}", "\"", "\\n", "<", ">", "|", "think", "tool"]
tags = list(literal._mask_of)
ok = True
for _ in range(300):
    pieces = ["".join(rng.choice(words) for _ in range(rng.randint(0, 12))) for _ in range(rng.randint(1, 6))]
    text = "".join(p + rng.choice(tags) for p in pieces)
    masked, _, applied = literal.mask([{"role": "user", "content": text}])
    s = masked[0]["content"]
    want = [i for part in literal._split.split(s) if part
            for i in (literal.plain[part] if part in literal.plain else tok.encode(part, add_special_tokens=False).ids)]
    ok &= applied and literal.encode(tok, s) == want and text_of(want) == text
check("4 300 random texts with tags: the batch encode equals encoding each piece, and decodes to the text", ok)


# 5. the server's own _prepare and tokenize
def app(cls, guard, kept=None):
    a = object.__new__(cls)
    a.engine = SimpleNamespace(supports_logprobs=False)
    a.vision = None
    a.served, a.aliases, a.model_dir = "GLM-5.3-Flash-EXL3", (), model_dir
    a.tok = tok
    a.template = ThinkingOffTemplate(template, kept) if cls is GlmApp else template
    if guard is not None:
        a.literal = guard
    a.default_thinking, a.reasoning_effort, a.thinking_budget = True, "high", 0
    a.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    a.max_tokens, a.native_context_window, a.context_window = 32768, 1048576, 1048576
    return a


kept = KeptReasoning()
patched, unpatched = app(GlmApp, literal, kept), app(GlmApp, None, kept)
body = lambda messages, **kw: {"model": "GLM-5.3-Flash-EXL3", "messages": messages, "max_tokens": 64, **kw}
plain = body([{"role": "user", "content": "こんにちは"}])
check("5 GlmApp, a plain chat: the unpatched ids", patched._prepare(plain, True).prompt ==
      unpatched._prepare(plain, True).prompt)
tagged = body([{"role": "user", "content": "Print as is: <think>hello</think> world"}])
p, u = patched._prepare(tagged, True).prompt, unpatched._prepare(tagged, True).prompt
check("5 GlmApp, quoted tags: the same text, one <think> token fewer",
      text_of(p) == text_of(u) and p.count(THINK) == u.count(THINK) - 1)
check("5 /tokenize agrees with the chat route", patched.tokenize({"messages": tagged["messages"]})["tokens"] == p)
shared = app(App, None)
s = shared._prepare(tagged, True).prompt
check("5 the shared App (other families): off, every quoted tag still a token",
      not hasattr(shared, "literal") and s.count(THINK) == text_of(s).count("<think>"))
described = body([{"role": "user", "content": "hi"}], tools=tools)
check("5 a tool described with tags: the reply parser gets the tools unmasked",
      patched._prepare(described, True).tools == unpatched._prepare(described, True).tools)

# kept reasoning: a client that drops it and renumbers the call ids, in a conversation that quotes a tag
history = [{"role": "user", "content": "make the test expect <think> first"}]
call = {"type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"cargo test\"}"}}
kept.remember(history, [{**call, "id": "call_0123456789abcdef01234567"}], "The output must open with <think>, so run it.")
later = body(history + [{"role": "assistant", "content": "", "tool_calls": [{**call, "id": "renumbered-1"}]},
                        {"role": "tool", "tool_call_id": "renumbered-1", "content": "1 passed"}], tools=TOOLS)
p, u = patched._prepare(later, True).prompt, unpatched._prepare(later, True).prompt
check("5 kept reasoning: put back for a client that renumbers ids, as without the patch",
      "The output must open with" in text_of(p) and text_of(p) == text_of(u))
check("5 kept reasoning: its quoted tag is text too", p.count(THINK) == u.count(THINK) - 2,
      f"<think> x{u.count(THINK)} -> x{p.count(THINK)} (the user's and the kept reasoning's: 2 fewer)")

os.environ["TF_GLM_LITERAL_TOKENS"] = "0"
check("5 TF_GLM_LITERAL_TOKENS=0: off", LiteralTokens.from_env(tok) is None)
del os.environ["TF_GLM_LITERAL_TOKENS"]
check("5 the startup line otherwise", LiteralTokens.from_env(tok) is not None)
chunk = "fn main() { println!(\"hello\"); } // 所有権と借用の説明。 " * 40
long = body([{"role": "user" if i % 2 == 0 else "assistant", "content": chunk + (" <think>" if i % 50 == 0 else "")}
             for i in range(300)])
t0 = time.perf_counter(); n = len(unpatched._prepare(long, True).prompt)
t1 = time.perf_counter(); patched._prepare(long, True)
t2 = time.perf_counter()
print(f"  a {n:,}-token history with 6 quoted tags: unpatched {t1 - t0:.2f} s, patched {t2 - t1:.2f} s")
check("5 a long history: about the same cost", t2 - t1 <= 1.5 * (t1 - t0) + 0.2)

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
