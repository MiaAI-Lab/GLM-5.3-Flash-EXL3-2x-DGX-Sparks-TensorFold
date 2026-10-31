#!/usr/bin/env python3
"""tools/reply_ids_check.py <checkpoint dir>: patch 0117, run in the image (CPU only, the server stopped or not).

GLM writes its markup (`</think>`, `<tool_call>`, `<arg_key>`, `<arg_value>` and their closers) as single tokens. The
same text spelled in ordinary tokens (a quoted tag, an example call, a file being written) used to be read as markup
by the server's text parsers. With the patch it is words. Checked through the CUDA server's chat route (`App.run`
behind `make_handler`, as the server runs it) on the checkpoint's own tokenizer:

1. A scripted engine writes each case's reply as token ids: text in ordinary tokens (the tokenizer with its added
   tokens left out, so a spelled tag is `</` `think` `>` and the like) and the marker tokens where the reply writes
   real markup. Each case runs whole and streamed, with the patch and without it (the old text parse).
2. The cases: a `</think>` quoted in the reasoning; an example call written as text in the answer; a call quoted in
   the reasoning before a real call; a real call whose argument spells `</arg_value></tool_call></think>`; a plain
   reply and a plain call. With the patch every case gives its expected reasoning, content and calls, the streamed
   deltas join to the whole reply, and no inert character reaches a client; a reply without spelled markup is the
   same with the patch as without it.
3. `GlmApp` turns the patch on, `TF_GLM_REPLY_IDS=0` turns it off, and the shared `App` (every other family) has it
   off.
Prints one line per case and N/N cases ok; exit 1 on any failure.

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/reply_ids_check.py:/c.py:ro" -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
      tensorfold-glm53:v0.6.0 /c.py \
      /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/<rev>
"""
import contextlib
import http.client
import inspect
import json
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from tokenizers import Tokenizer

from tensorfold.cuda import server
from tensorfold.cuda.reply_text import ReplyMarks

model = sys.argv[1]
tok = Tokenizer.from_file(f"{model}/tokenizer.json")
spec = json.loads(Path(f"{model}/tokenizer.json").read_text(encoding="utf-8"))
spec["added_tokens"] = []
plain = Tokenizer.from_str(json.dumps(spec))
ID = {m: tok.token_to_id(m) for m in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
                                      "<arg_value>", "</arg_value>", "<|user|>")}
assert all(v is not None for v in ID.values()), ID
EOS = ID["<|user|>"]
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def ids(*parts):
    """Strings as ordinary tokens; ("<tool_call>",) and the like as the marker token."""

    out = []
    for p in parts:
        out += [ID[p[0]]] if isinstance(p, tuple) else plain.encode(p, add_special_tokens=False).ids
    return out


def real_call(command):
    return [("<tool_call>",), "terminal", ("<arg_key>",), "command", ("</arg_key>",), ("<arg_value>",), command,
            ("</arg_value>",), ("</tool_call>",)]


SPELLED_CALL = "<tool_call>terminal<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
CASES = [
    # name, tools, reply parts, expected reasoning, content, calls [(name, arguments)]
    ("quoted </think> in reasoning, echoed tags in the answer", False,
     ["The user wants <think>hello</think> world back.", ("</think>",), "<think>hello</think> world"],
     "The user wants <think>hello</think> world back.", "<think>hello</think> world", None),
    ("an example call written as text: words, not a call", True,
     ["Explain the format.", ("</think>",), "A call looks like " + SPELLED_CALL + " in GLM."],
     "Explain the format.", "A call looks like " + SPELLED_CALL + " in GLM.", None),
    ("a call quoted in the reasoning, a real call after the close", True,
     ["I could write " + SPELLED_CALL + " myself.", ("</think>",), *real_call("ls -la")],
     "I could write " + SPELLED_CALL + " myself.", "", [("terminal", {"command": "ls -la"})]),
    ("a real call whose argument spells the closers", True,
     ["Write the file.", ("</think>",), *real_call("echo '</arg_value></tool_call></think>' > x")],
     "Write the file.", "", [("terminal", {"command": "echo '</arg_value></tool_call></think>' > x"})]),
    ("plain reply: the same as without the patch", True,
     ["Plan.", ("</think>",), "Done."], "Plan.", "Done.", None),
    ("plain call: the same as without the patch", True,
     ["Run it.", ("</think>",), *real_call("ls")], "Run it.", "", [("terminal", {"command": "ls"})]),
]


class Engine:
    eos = (EOS,)

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        for at in range(0, len(self.reply), 3):
            if on_tokens(self.reply[at:at + 3]):
                break
        return {"rounds": 1 + len(self.reply) // 3, "decode_s": 0.1}


TEMPLATE = ("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}{% if tools %}tools;{% endif %}"
            "assistant:{% if enable_thinking %}<think>{% endif %}")


def app_for(reply, patched, tmp):
    (tmp / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(reply + [EOS]), "glm-check", tok
    app.template = server.ChatTemplate(tmp)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.reply_ids = ReplyMarks(tok) if patched else None
    return app


@contextlib.contextmanager
def http_server(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()


def ask(app, tools, stream):
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": stream}
    if tools:
        body["tools"] = TOOLS
    with http_server(app) as port:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        text = r.read().decode()
        c.close()
    assert r.status == 200, (r.status, text[:300])
    if not stream:
        m = json.loads(text)["choices"][0]["message"]
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]
        return m.get("reasoning_content") or "", m.get("content") or "", calls or None, text
    reasoning, content, calls = "", "", {}
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices") or []:
            d = ch.get("delta") or {}
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            for c in d.get("tool_calls") or []:
                slot = calls.setdefault(c.get("index", 0), ["", ""])
                fn = c.get("function") or {}
                slot[0] += fn.get("name") or ""
                slot[1] += fn.get("arguments") or ""
    got = [(n, json.loads(a)) for n, a in (calls[k] for k in sorted(calls))]
    return reasoning, content, got or None, text


fails = 0
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    for name, tools, parts, want_r, want_c, want_calls in CASES:
        reply = ids(*parts)
        want = (want_r, want_c, want_calls)
        rows = {}
        for patched in (True, False):
            for stream in (False, True):
                r, c, calls, raw = ask(app_for(reply, patched, tmp), tools, stream)
                leaked = chr(0xE000) in raw or chr(92) + "ue000" in raw
                rows[(patched, stream)] = ((r.strip(), c.strip(), calls), leaked)
        good = all(rows[(True, s)][0] == want and not rows[(True, s)][1] for s in (False, True))
        spelled = rows[(False, False)][0] != want
        if name.startswith("plain"):
            good = good and not spelled
        fails += not good
        print(f"{'ok  ' if good else 'FAIL'} {name}")
        for (patched, stream), (got, leak) in rows.items():
            mark = "" if got == want else ("  <- the bug" if not patched else "  <- WRONG")
            print(f"     {'0117' if patched else 'text'} {'stream' if stream else 'whole '}: reasoning={got[0]!r} "
                  f"content={got[1]!r} calls={got[2]!r}{'  INERT LEAKED' if leak else ''}{mark}")

# the switch: GlmApp turns it on from the environment, 0 turns it off, every other family's App has none
from tensorfold.families.glm5_next.cuda.app import GlmApp  # noqa: E402

on = ReplyMarks.from_env(tok)
os.environ["TF_GLM_REPLY_IDS"] = "0"
off = ReplyMarks.from_env(tok)
del os.environ["TF_GLM_REPLY_IDS"]
switch = (on is not None and len(on.ids) == 8 and off is None
          and "ReplyMarks.from_env(self.tok)" in inspect.getsource(GlmApp.__init__)
          and getattr(server.App.__new__(server.App), "reply_ids", None) is None)
fails += not switch
print(f"{'ok  ' if switch else 'FAIL'} the switch: GlmApp on ({len(on.ids) if on else 0} marker tokens), "
      f"TF_GLM_REPLY_IDS=0 off, other families' App off")
print(f"{len(CASES) + 1 - fails}/{len(CASES) + 1} cases ok")
sys.exit(1 if fails else 0)
