#!/usr/bin/env python3
"""tools/stream_tool_calls_check.py <checkpoint dir>: patch 0132, run in the image (CPU only, server stopped or not).

GLM-5.3 writes a tool call as `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`. Before
0132 a streamed reply held each call and sent it whole at its `</tool_call>`, so a long Write sent no argument for
minutes. With 0132 the call's header goes out when its name is whole, a string value goes out in JSON-escaped pieces
while it decodes, and the pieces join to the whole reply's arguments byte for byte. Checked through the CUDA server's
chat route and its `/v1/messages` route (`App.run` behind `make_handler`, as the server runs it) on the checkpoint's
own tokenizer and template, with a scripted engine that writes each case's reply as GLM-5.3's token ids in decode
rounds of given sizes:

1. 22 cases, each whole and streamed on both routes (and with STREAM_SMOOTH on for some): a long Write, an Edit, two
   calls, prose before and after a call, typed values (array, null, issue #180's union, integers, an unknown key),
   CJK and emoji split inside a character, a value that spells the closers, quoted calls, a call cut by max_tokens or
   by a stop string, issue #22's two shapes, a repeated key, an unoffered tool, newlines between the tags, calls in
   the think block, the one-call policy, and replies without tools or without a call.
2. Every whole reply, and every stream of a case that sends no call while it decodes, hashes to what the engine
   before 0132 gave (ids and `created` left out; `GOLDEN`, recorded with --print-golden on main's patches).
3. Each stream joins to its whole reply: reasoning, content, finish reason, and each closed call's name and arguments
   byte for byte; a cut call is never closed. Each header comes once and in order; nothing but a call's own pieces
   comes between its header and its last piece; no inert U+E000 (0117) reaches a client. A long string value comes
   in at least 10 argument deltas on the chat stream and as at least 10 `input_json_delta` on `/v1/messages`.
4. `GlmCallStreamer` alone: a spelled tag split between two pieces comes back whole; a text that does not extend the
   one read turns it off; a value streams and its brace waits for `</tool_call>`; a tail that may begin a closer waits.
5. --fuzz N random replies (default 20000): `GlmCallStreamer` fed each at random split points against
   `parse_tool_calls` on the whole text. Every call it sends closed is one the parser returns, in order; a well-formed
   reply's closed calls are all of the parser's; a long string value comes in more than one piece.
--expect-stock, on the patches before 0132: exits 0 when only the streaming checks fail (the long calls arrive whole)
and every hash matches. Prints one line per case and N/N ok; exit 1 on any failure.

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/stream_tool_calls_check.py:/c.py:ro" -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
      tensorfold-glm53:v0.6.0 /c.py \
      /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/<rev>

On a host (tokenizers and jinja2 installed): add --source-root DIR, a TensorFold v0.6.0 src/ with the patches applied.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import importlib
import io
import json
import os
import random
import re
import sys
import threading
import types
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

MARKS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
         "<|observation|>", "<|user|>", "<|endoftext|>")
EOS = "<|observation|>"          # what GLM writes after its calls (one of generation_config's three end tokens)
END = "<|user|>"                 # what it writes after an answer


def fn(name: str, props: dict[str, Any], required: list[str] | None = None, description: str = "") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description or name, "parameters": {
        "type": "object", "properties": props, "required": required or []}}}


# An agent client's file and shell tools (their parameter shapes), plus the schema kinds the end parser types otherwise
WRITE = fn("Write", {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"])
EDIT = fn("Edit", {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
                   "replace_all": {"type": "boolean", "default": False}}, ["file_path", "old_string", "new_string"])
BASH = fn("Bash", {"command": {"type": "string"}, "timeout": {"type": "number"}, "description": {"type": "string"},
                   "run_in_background": {"type": "boolean"}}, ["command"])
READ = fn("Read", {"file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
          ["file_path"])
TODO = fn("TodoWrite", {"todos": {"type": "array", "items": {"type": "object", "properties": {
    "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
    "activeForm": {"type": "string"}}}}}, ["todos"])
FETCH = fn("WebFetch", {"url": {"type": "string"}, "prompt": {"type": ["string", "null"]}}, ["url"])
CONFIG = fn("config_set", {"key": {"type": "string"}, "value": {"type": ["object", "string"]}}, ["key", "value"])
ANY = fn("record", {"payload": {}}, ["payload"])
TOOLS = [WRITE, EDIT, BASH, READ, TODO, FETCH, CONFIG, ANY]

WRITE_PATH = "/home/user/project/src/ratelimit.py"
WRITE_CONTENT = '''"""A token bucket: at most `rate` requests a second, bursts up to `burst`."""
from __future__ import annotations

import re
import threading
import time

_KEY = re.compile(r"^[a-z0-9_.:-]{1,64}$")


class TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError(f"rate={rate!r} and burst={burst!r} must be positive")
        self.rate, self.burst = rate, burst
        self.tokens = float(burst)
        self.stamp = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: int = 1) -> bool:
        """True when `n` tokens were taken; False → the caller waits (no partial takes)."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.stamp) * self.rate)
            self.stamp = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


def bucket_key(user: str) -> str:
    if not _KEY.match(user):
        raise KeyError("bad key: %s" % user.replace("\\n", "\\\\n"))
    return "rl:" + user
'''

UNICODE_CONTENT = ("# 速率限制器 — 说明\n\n每个用户每秒最多 `rate` 个请求，突发上限 `burst`。🚦🚀\n\n"
                   "| 参数 | 含义 |\n|---|---|\n| rate | 每秒令牌数 ✅ |\n| burst | 桶容量 🪣 |\n\n"
                   "Ünïcödé: naïve café, Ελληνικά, русский, עברית, 日本語のテキスト。👩‍💻👨🏽‍🚀\n")

TODOS = [{"content": "Write the rate limiter", "status": "completed", "activeForm": "Writing the rate limiter"},
         {"content": "Add tests for bursts", "status": "in_progress", "activeForm": "Adding tests for bursts"},
         {"content": "Wire it into the API", "status": "pending", "activeForm": "Wiring it into the API"}]


def M(mark: str) -> tuple[str]:
    """A marker token."""
    return (mark,)


def call(name: str, args: list[tuple[str, str]], newlines: bool = False) -> list[Any]:
    """GLM's markup for one call: each value as written (text as is, other values as JSON text)."""

    nl = "\n" if newlines else ""
    parts: list[Any] = [M("<tool_call>"), name + nl]
    for k, v in args:
        parts += [M("<arg_key>"), k, M("</arg_key>")] + ([nl] if nl else []) + [M("<arg_value>"), v, M("</arg_value>")]
        if nl:
            parts.append(nl)
    parts.append(M("</tool_call>"))
    return [p for p in parts if p != ""]


@dataclass
class Case:
    name: str
    parts: list[Any]                       # the reply: strings (ordinary tokens) and marker tuples
    tools: list[dict[str, Any]] | None = field(default_factory=lambda: TOOLS)
    thinking: bool = True
    rounds: list[int] = field(default_factory=lambda: [3])     # tokens a decode round, cycled
    max_tokens: int | None = None
    body: dict[str, Any] = field(default_factory=dict)        # more request fields (stop, parallel_tool_calls, ...)
    smooth: bool = False                   # also stream it with TF_GLM_STREAM_SMOOTH=1
    unchanged: bool = False                # its stream must equal the stock engine's (no call streams)
    write_like: bool = False               # a long string value: its arguments must arrive in >= 10 deltas
    note: str = ""


BASH_LS = call("Bash", [("command", "ls -la src/ && git status --short"), ("description", "List the source tree")])
READ_CFG = call("Read", [("file_path", "/home/user/project/pyproject.toml"), ("limit", "80")])
SPELLED = "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"

CASES: list[Case] = [
    Case("write_long", ["The user wants a rate limiter module. I'll write it to src/ratelimit.py with a token bucket "
                        "and a key check.", M("</think>"), "\n",
                        *call("Write", [("file_path", WRITE_PATH), ("content", WRITE_CONTENT)]), M(EOS)],
         rounds=[3, 1, 4, 2, 6, 3], smooth=True, write_like=True, note="one long Write, as an agent writes a file"),
    Case("edit", ["Fix the off-by-one.", M("</think>"),
                  *call("Edit", [("file_path", WRITE_PATH), ("old_string", "if self.tokens >= n:\n"),
                                 ("new_string", "if self.tokens + 1e-9 >= n:\n"), ("replace_all", "false")]), M(EOS)]),
    Case("parallel_two", ["Read the config and list the tree.", M("</think>"), *READ_CFG, "\n", *BASH_LS, M(EOS)],
         rounds=[2, 5], smooth=True),
    Case("prose_then_call", ["I'll check the directory first.\n", *BASH_LS, M(EOS)], thinking=False, rounds=[4]),
    Case("typed_values", ["Track the work, then set the cache policy.", M("</think>"),
                          *call("TodoWrite", [("todos", json.dumps(TODOS, ensure_ascii=False))]),
                          *call("WebFetch", [("url", "https://example.com/docs"), ("prompt", "null")]),
                          *call("config_set", [("key", "cache.policy"),
                                               ("value", '{"tiers": ["mem", "disk"], "ttl": 300}')]),
                          *call("Read", [("file_path", "/etc/hosts"), ("offset", "10"), ("limit", "5"),
                                         ("verbose", "yes")]), M(EOS)],
         rounds=[5], note="array, null, #180 union, integers, an unknown key"),
    Case("unicode_write", ["写一个说明文件。", M("</think>"),
                           *call("Write", [("file_path", "/home/user/project/README.zh.md"),
                                           ("content", UNICODE_CONTENT)]), M(EOS)],
         rounds=[1, 2, 1, 3], write_like=True, note="CJK, emoji, ZWJ sequences: tokens that end inside a character"),
    Case("spelled_closers", ["Write the marker file.", M("</think>"),
                             *call("Bash", [("command", "printf '%s\\n' '</arg_value></tool_call></think>' > m.txt")]),
                             M(EOS)], note="a value that spells the closers in ordinary tokens"),
    Case("quoted_calls", ["I could write " + SPELLED + " myself, but the real call is better.", M("</think>"),
                          "A call looks like " + SPELLED + " in GLM.\n", *BASH_LS, M(EOS)],
         note="a spelled call in the reasoning and in the text: no call; the real one after"),
    Case("cut_by_max_tokens", ["Write it.", M("</think>"),
                               *call("Write", [("file_path", WRITE_PATH), ("content", WRITE_CONTENT)]), M(EOS)],
         max_tokens=120, note="the token limit cuts the content value"),
    Case("end_before_closer", ["List it.", M("</think>"), M("<tool_call>"), "Bash", M("<arg_key>"), "command",
                               M("</arg_key>"), M("<arg_value>"), "ls -la", M("</arg_value>"), M(EOS)],
         note="#22: the end token before </tool_call>; the server closes it"),
    Case("one_call_policy", ["Write it.", M("</think>"),
                             *call("Write", [("file_path", WRITE_PATH), ("content", WRITE_CONTENT)]), M(EOS)],
         body={"parallel_tool_calls": False}, unchanged=True, note="parallel_tool_calls false: one call, at the end"),
    Case("no_tools", ["Say hi.", M("</think>"), "Hello! How can I help with the project today?", M(END)],
         tools=None, unchanged=True, smooth=True),
    Case("no_tools_thinking_off", ["Hello! Here is a short answer with <tool_call> in it, as text.", M(END)],
         tools=None, thinking=False, unchanged=True),
    Case("tools_no_call", ["Nothing to run.", M("</think>"), "The tree is already clean; no tool is needed.", M(END)],
         unchanged=True),
    Case("repeated_key", ["Run it.", M("</think>"),
                          *call("Bash", [("command", "ls"), ("description", "list"), ("command", "ls -la")]), M(EOS)],
         note="a repeated key"),
    Case("missing_arg_key", ["Read it.", M("</think>"), M("<tool_call>"), "Read", M("<arg_key>"), "file_path",
                             M("</arg_key>"), M("<arg_value>"), "/etc/hosts", M("</arg_value>"), "\nlimit",
                             M("</arg_key>"), M("<arg_value>"), "5", M("</arg_value>"), M("</tool_call>"), M(EOS)],
         note="#22: an <arg_key> written as whitespace; the parser puts it back"),
    Case("unoffered_tool", ["Use the agent tool.", M("</think>"),
                            *call("spawn_agent", [("task", "review")]), M(EOS)], note="stays text"),
    Case("prose_after_call", ["Run then report.", M("</think>"), *BASH_LS, "Started the listing.", M(END)],
         rounds=[40], smooth=True, note="one round holds the closer and the prose after it"),
    Case("newline_style", ["List it.", M("</think>"), *call("Bash", [("command", "ls -la"), ("timeout", "30000")],
                                                           newlines=True), M(EOS)], note="GLM-4.x newlines"),
    Case("stop_in_value", ["Write it.", M("</think>"),
                           *call("Write", [("file_path", WRITE_PATH), ("content", WRITE_CONTENT)]), M(EOS)],
         body={"stop": ["self.lock"]}, note="a stop string inside the value"),
    Case("call_in_think_block", ["Just list the tree. ", *BASH_LS, M("</think>"), M(EOS)],
         note="a short call that ends the reply inside the think block: the end parser's, whole"),
    Case("long_call_in_think_block", ["Writing now. ",
                                      *call("Write", [("file_path", WRITE_PATH), ("content", WRITE_CONTENT * 2)]),
                                      M("</think>"), M(EOS)],
         write_like=True, note="a call past THINK_CALL_HOLD inside the think block: committed, streamed"),
]


class Glm:
    """The checkpoint's tokenizer, as the cases need it."""

    def __init__(self, tok: Any) -> None:
        from tokenizers import Tokenizer

        self.tok = tok
        spec = json.loads(tok.to_str())
        spec["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(spec))
        self.id = {m: tok.token_to_id(m) for m in MARKS}
        missing = [m for m, i in self.id.items() if i is None]
        if missing:
            raise ValueError(f"not GLM-5.3's tokenizer: no {missing}")

    def ids(self, parts: list[Any]) -> list[int]:
        out: list[int] = []
        for p in parts:
            out += [self.id[p[0]]] if isinstance(p, tuple) else self.plain.encode(p, add_special_tokens=False).ids
        return out


GLM_EOS = (154820, 154827, 154829)       # generation_config.json's end tokens: <|endoftext|>, <|user|>, <|observation|>


def load_engine(source_root: str | None):
    """TensorFold from ``source_root`` (which holds ``tensorfold/``), or the installed one (the image's) when None."""

    if source_root is None:
        return tuple(importlib.import_module(f"tensorfold.{name}")
                     for name in ("cuda.server", "cuda.reply_text", "families.glm5_next.cuda.app"))
    root = str(Path(source_root).resolve())
    if not (Path(root) / "tensorfold" / "cuda" / "server.py").exists():
        raise SystemExit(f"{root} holds no tensorfold/cuda/server.py")
    for name in [m for m in sys.modules if m == "tensorfold" or m.startswith("tensorfold.")]:
        del sys.modules[name]
    sys.path.insert(0, root)
    try:
        server = importlib.import_module("tensorfold.cuda.server")
        reply_text = importlib.import_module("tensorfold.cuda.reply_text")
        glm = importlib.import_module("tensorfold.families.glm5_next.cuda.app")
    finally:
        sys.path.remove(root)
    return server, reply_text, glm


class Engine:
    """One scripted reply a request: ``script`` is set before each request (the server runs one at a time here)."""

    eos = GLM_EOS

    def __init__(self) -> None:
        self.script: list[int] = []
        self.rounds: list[int] = [3]
        self.request = types.SimpleNamespace(policy=None, stop_eos=True)
        self.sent_rounds: list[list[int]] = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True, **_):
        self.sent_rounds = []
        at, k = 0, 0
        while at < len(self.script) and at < max_tokens:
            n = max(1, self.rounds[k % len(self.rounds)])
            k += 1
            chunk = self.script[at:min(at + n, max_tokens)]
            cut = next((i + 1 for i, t in enumerate(chunk) if stop_eos and t in self.eos), len(chunk))
            chunk = chunk[:cut]
            at += len(chunk)
            self.sent_rounds.append(chunk)
            if on_tokens(chunk) or (chunk and stop_eos and chunk[-1] in self.eos):
                break
        return {"rounds": k, "decode_s": 0.25, "prompt_s": 0.01}


class Harness:
    def __init__(self, source_root: str | None, checkpoint: str | Path) -> None:
        from tokenizers import Tokenizer

        self.server, self.reply_text, self.glm = load_engine(source_root)
        self.tok = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
        template_dir = Path(checkpoint)
        self.engine = Engine()
        app = self.glm.GlmApp.__new__(self.glm.GlmApp)
        app.engine, app.vision, app.served, app.aliases = self.engine, None, "GLM-5.3-Flash-EXL3", ()
        app.model_dir = Path(template_dir)
        app.tok = self.tok
        app.template = self.server.ChatTemplate(Path(template_dir))
        app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
        app.sampling, app.max_tokens = {"temperature": 0.0}, 32768
        app.native_context_window, app.context_window = 1048576, 0
        app.reply_ids = self.reply_text.ReplyMarks(self.tok)
        app.literal = None
        self.app = app
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.server.make_handler(app))
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.lock = threading.Lock()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- requests ------------------------------------------------------------------------------------------------
    def _post(self, path: str, body: dict[str, Any]) -> tuple[int, str]:
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=120)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read().decode("utf-8")
        finally:
            c.close()

    def run(self, script: list[int], rounds: list[int], path: str, body: dict[str, Any],
            smooth: bool = False) -> tuple[int, str]:
        with self.lock:
            self.engine.script, self.engine.rounds = list(script), list(rounds)
            env = {"TF_GLM_STREAM_SMOOTH": "1" if smooth else "0", "TF_GLM_STREAM_SMOOTH_MS": "20"}
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                return self._post(path, body)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


def chat_body(tools: list[dict[str, Any]] | None, thinking: bool, stream: bool, max_tokens: int | None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "GLM-5.3-Flash-EXL3", "stream": stream, "temperature": 0,
                            "messages": [{"role": "user", "content": "Please do the task."}],
                            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body["tools"] = tools
    if max_tokens:
        body["max_tokens"] = max_tokens
    body.update(extra or {})
    return body


def messages_body(tools: list[dict[str, Any]] | None, thinking: bool, stream: bool, max_tokens: int | None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "GLM-5.3-Flash-EXL3", "stream": stream, "temperature": 0,
                            "max_tokens": max_tokens or 32000,
                            "messages": [{"role": "user", "content": "Please do the task."}],
                            "thinking": {"type": "adaptive" if thinking else "disabled"}}
    if tools:
        body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                          "input_schema": t["function"]["parameters"]} for t in tools]
    extra = dict(extra or {})
    if "stop" in extra:
        body["stop_sequences"] = extra.pop("stop")
    if extra.pop("parallel_tool_calls", True) is False:
        body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    body.update(extra)
    return body


# -- reading replies --------------------------------------------------------------------------------------------------
def chat_events(text: str) -> list[dict[str, Any]]:
    """A chat stream's data events (keep-alive comments and [DONE] left out)."""

    out = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def messages_events(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if data:
            out.append(json.loads("\n".join(data)))
    return out


class ChatStream:
    """What a chat stream says: joined reasoning and content, each call's header and argument pieces, the order of
    everything (for the order rules), the finish reason."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.reasoning = self.content = ""
        self.calls: dict[int, dict[str, Any]] = {}
        self.order: list[tuple[str, int | None]] = []        # ("text"|"head"|"args"|"empty", index)
        self.finish = None
        self.errors: list[str] = []
        for ev in events:
            if "error" in ev:
                self.errors.append(json.dumps(ev["error"]))
                continue
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    self.reasoning += d["reasoning_content"]
                    self.order.append(("reasoning", None))
                if d.get("content"):
                    self.content += d["content"]
                    self.order.append(("text", None))
                for c in d.get("tool_calls") or []:
                    i = c.get("index", 0)
                    fn = c.get("function") or {}
                    slot = self.calls.setdefault(i, {"id": None, "name": "", "arguments": "", "pieces": 0, "heads": 0})
                    if c.get("id"):
                        slot["id"] = c["id"]
                        slot["heads"] += 1
                        slot["name"] += fn.get("name") or ""
                        self.order.append(("head", i))
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                        slot["pieces"] += 1
                        self.order.append(("args", i))
                if not d and not ch.get("finish_reason"):
                    self.order.append(("empty", None))

    def joined(self) -> list[tuple[str, str]]:
        return [(c["name"], c["arguments"]) for _, c in sorted(self.calls.items())]

    def closed(self) -> list[tuple[str, str]]:
        """The calls whose arguments are a whole JSON object (sent closed)."""

        out = []
        for name, args in self.joined():
            try:
                if isinstance(json.loads(args), dict):
                    out.append((name, args))
            except ValueError:
                pass
        return out

    def order_problems(self) -> list[str]:
        """Each header once, in index order, before its pieces; nothing but a call's own pieces (and empty keep-alive
        deltas) between its header and its last piece, for a call sent closed."""

        problems = []
        heads = [i for kind, i in self.order if kind == "head"]
        if heads != sorted(heads) or len(heads) != len(set(heads)):
            problems.append(f"headers out of order or repeated: {heads}")
        closed = {i for i, c in self.calls.items() if (c["name"], c["arguments"]) in self.closed()}
        for i, c in self.calls.items():
            if c["heads"] != 1:
                problems.append(f"call {i}: {c['heads']} headers")
            marks = [n for n, (kind, j) in enumerate(self.order) if j == i]
            if not marks or self.order[marks[0]][0] != "head":
                problems.append(f"call {i}: a piece before its header")
                continue
            if i in closed:
                between = {self.order[n][0] for n in range(marks[0], marks[-1] + 1)
                           if self.order[n][1] != i and self.order[n][0] != "empty"}
                if between:
                    problems.append(f"call {i}: {sorted(between)} between its header and its last piece")
        return problems


class Whole:
    def __init__(self, text: str) -> None:
        payload = json.loads(text)
        self.payload = payload
        m = payload["choices"][0]["message"]
        self.reasoning = m.get("reasoning_content") or ""
        self.content = m.get("content") or ""
        self.calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
        self.finish = payload["choices"][0]["finish_reason"]


class MessagesStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.blocks: dict[int, dict[str, Any]] = {}
        self.stop_reason = None
        self.errors: list[str] = []
        for ev in events:
            kind = ev.get("type")
            if kind == "content_block_start":
                self.blocks[ev["index"]] = {**ev["content_block"], "json": "", "deltas": 0, "text_deltas": 0}
            elif kind == "content_block_delta":
                b, d = self.blocks[ev["index"]], ev["delta"]
                if d["type"] == "input_json_delta":
                    b["json"] += d["partial_json"]
                    b["deltas"] += 1
                elif d["type"] == "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                    b["text_deltas"] += 1
                elif d["type"] == "thinking_delta":
                    b["thinking"] = b.get("thinking", "") + d["thinking"]
            elif kind == "message_delta":
                self.stop_reason = ev["delta"].get("stop_reason")
            elif kind == "error":
                self.errors.append(json.dumps(ev))

    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for _, b in sorted(self.blocks.items()) if b["type"] == "tool_use"]


_IDS = re.compile(r'"(?:chatcmpl|cmpl|call|msg|toolu)[-_][0-9a-zA-Z]+"')
_CREATED = re.compile(r'"created": \d+')


def normalized(text: str) -> str:
    """A reply with its random ids and its time left out, for byte comparisons across runs and engines."""

    return _CREATED.sub('"created": 0', _IDS.sub('"<id>"', text))


J = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))  # noqa: E731  the parser's JSON

EXPECT = {
    "write_long": {"calls": [("Write", J({"file_path": WRITE_PATH, "content": WRITE_CONTENT}))],
                   "finish": "tool_calls"},
    "edit": {"calls": [("Edit", J({"file_path": WRITE_PATH, "old_string": "if self.tokens >= n:\n",
                                   "new_string": "if self.tokens + 1e-9 >= n:\n", "replace_all": False}))]},
    "parallel_two": {"calls": [("Read", J({"file_path": "/home/user/project/pyproject.toml", "limit": 80})),
                               ("Bash", J({"command": "ls -la src/ && git status --short",
                                           "description": "List the source tree"}))]},
    "typed_values": {"calls": [("TodoWrite", J({"todos": TODOS})),
                               ("WebFetch", J({"url": "https://example.com/docs", "prompt": None})),
                               # kit issue #180: a union with "string" keeps the text (both paths, unchanged)
                               ("config_set", J({"key": "cache.policy",
                                                 "value": '{"tiers": ["mem", "disk"], "ttl": 300}'})),
                               ("Read", J({"file_path": "/etc/hosts", "offset": 10, "limit": 5, "verbose": "yes"}))]},
    "unicode_write": {"calls": [("Write", J({"file_path": "/home/user/project/README.zh.md",
                                             "content": UNICODE_CONTENT}))]},
    "spelled_closers": {"calls": [("Bash", J({"command":
                                              "printf '%s\\n' '</arg_value></tool_call></think>' > m.txt"}))]},
    "quoted_calls": {"calls": [("Bash", J({"command": "ls -la src/ && git status --short",
                                           "description": "List the source tree"}))],
                     "content": "A call looks like " + SPELLED + " in GLM."},
    "cut_by_max_tokens": {"calls": [], "finish": "length", "open": 1},
    "end_before_closer": {"calls": [("Bash", J({"command": "ls -la"}))], "finish": "tool_calls"},
    "repeated_key": {"calls": [("Bash", J({"command": "ls -la", "description": "list"}))],
                     "stream_args": ['{"command":"ls","description":"list","command":"ls -la"}']},
    "missing_arg_key": {"calls": [("Read", J({"file_path": "/etc/hosts", "limit": 5}))]},
    "unoffered_tool": {"calls": [], "finish": "stop"},
    "prose_after_call": {"calls": [("Bash", J({"command": "ls -la src/ && git status --short",
                                               "description": "List the source tree"}))],
                         "content": "Started the listing.", "text_after_call": True},
    "newline_style": {"calls": [("Bash", J({"command": "ls -la", "timeout": 30000}))]},
    "stop_in_value": {"calls": [], "finish": "stop", "open": 1},
    "call_in_think_block": {"calls": [("Bash", J({"command": "ls -la src/ && git status --short",
                                                  "description": "List the source tree"}))]},
    "long_call_in_think_block": {"calls": [("Write", J({"file_path": WRITE_PATH,
                                                        "content": WRITE_CONTENT * 2}))]},
}
# streams that send no call while it decodes: they must equal the stock engine's
UNCHANGED_STREAMS = {c.name for c in CASES if c.unchanged} | {"call_in_think_block"}


def routes(case: Case) -> list[tuple[str, str, bool, bool]]:
    """(route label, path, stream, smooth)."""

    out = [("chat-whole", "/v1/chat/completions", False, False), ("chat-stream", "/v1/chat/completions", True, False),
           ("messages-whole", "/v1/messages", False, False), ("messages-stream", "/v1/messages", True, False)]
    if case.smooth:
        out.insert(2, ("chat-stream-smooth", "/v1/chat/completions", True, True))
    return out


def run_case(h: Harness, case: Case, ids: list[int]) -> dict[str, str]:
    raw = {}
    for label, path, stream, smooth in routes(case):
        make = chat_body if path.endswith("completions") else messages_body
        body = make(case.tools, case.thinking, stream, case.max_tokens, case.body)
        status, text = h.run(ids, case.rounds, path, body, smooth=smooth)
        if status != 200:
            raise AssertionError(f"{label}: HTTP {status}: {text[:300]}")
        raw[label] = text
    return raw


def check(case: Case, raw: dict[str, str]) -> tuple[list[str], list[str]]:
    """(problems, streaming problems): the second are the ones the stock engine has by design."""

    problems, streaming = [], []
    want = EXPECT.get(case.name, {})
    whole = Whole(raw["chat-whole"])
    for label in [r for r in raw if r.startswith("chat-stream")]:
        s = ChatStream(chat_events(raw[label]))
        if s.errors:
            problems.append(f"{label}: error events {s.errors}")
        if (s.reasoning.strip(), s.content.strip(), s.finish) != (whole.reasoning.strip(), whole.content.strip(),
                                                                   whole.finish):
            problems.append(f"{label}: joined reasoning/content/finish {(s.reasoning[-60:], s.content[-60:], s.finish)}"
                            f" differ from the whole reply's "
                            f"{(whole.reasoning[-60:], whole.content[-60:], whole.finish)}")
        closed = s.closed()
        if want.get("stream_args"):
            if [a for _, a in closed] != want["stream_args"]:
                problems.append(f"{label}: streamed arguments {[a for _, a in closed]}")
            if [json.loads(a) for _, a in closed] != [json.loads(a) for _, a in whole.calls]:
                problems.append(f"{label}: streamed calls decode otherwise than the whole reply's")
        elif closed != whole.calls:
            problems.append(f"{label}: closed calls {closed} differ from the whole reply's {whole.calls}")
        opened = len(s.calls) - len(closed)
        if opened != want.get("open", 0):            # by design: before 0132 a cut call never showed
            streaming.append(f"{label}: {opened} calls left open (want {want.get('open', 0)})")
        problems += [f"{label}: {p}" for p in s.order_problems()]
        if want.get("text_after_call"):
            last_piece = max(n for n, (kind, i) in enumerate(s.order) if kind in ("head", "args"))
            if not any(kind == "text" for kind, _ in s.order[last_piece:]):      # the stock engine's order
                streaming.append(f"{label}: the prose after the call came before the call's last piece")
        if case.write_like:
            longest = max((c["pieces"] for c in s.calls.values()), default=0)
            if longest < 10:
                streaming.append(f"{label}: the long call came in {longest} argument deltas (want >= 10)")
    mwhole = json.loads(raw["messages-whole"])
    ms = MessagesStream(messages_events(raw["messages-stream"]))
    if ms.errors:
        problems.append(f"messages-stream: error events {ms.errors}")
    whole_inputs = [(b["name"], b["input"]) for b in mwhole["content"] if b["type"] == "tool_use"]
    streamed_inputs = []
    for b in ms.tool_uses():
        try:
            streamed_inputs.append((b["name"], json.loads(b["json"] or "{}")))
        except ValueError:
            if not want.get("open"):
                problems.append(f"messages-stream: tool_use {b['name']} input is not JSON: {b['json'][-80:]!r}")
    if not want.get("open") and streamed_inputs != whole_inputs:
        problems.append(f"messages-stream: tool inputs {streamed_inputs} differ from the whole reply's {whole_inputs}")
    if ms.stop_reason != mwhole["stop_reason"]:
        problems.append(f"messages-stream: stop_reason {ms.stop_reason} vs {mwhole['stop_reason']}")
    if [J(i) for _, i in whole_inputs] != [J(json.loads(a)) for _, a in whole.calls]:
        problems.append("messages-whole: inputs differ from the chat reply's calls")
    if case.write_like:
        deltas = max((b["deltas"] for b in ms.tool_uses()), default=0)
        if deltas < 10:
            streaming.append(f"messages-stream: the long call came in {deltas} input_json_delta events (want >= 10)")
        for b in ms.tool_uses():
            joined = b["json"]
            wanted = [a for n, a in whole.calls if n == b["name"]]
            if wanted and joined != wanted[0]:
                problems.append(f"messages-stream: the joined input_json_delta of {b['name']} is not the chat reply's "
                                f"arguments byte for byte")
    if "calls" in want and whole.calls != want["calls"]:
        problems.append(f"chat-whole: calls {whole.calls} (want {want['calls']})")
    if "finish" in want and whole.finish != want["finish"]:
        problems.append(f"chat-whole: finish {whole.finish} (want {want['finish']})")
    if "content" in want and whole.content.strip() != want["content"]:
        problems.append(f"chat-whole: content {whole.content!r}")
    for label, text in raw.items():
        if chr(0xE000) in text or "\\ue000" in text:
            problems.append(f"{label}: an inert marker reached the client")
    return problems, streaming


def streamer_checks(h: Harness) -> list[tuple[str, list[str]]]:
    """``GlmCallStreamer`` alone, on the corners the routes reach rarely (each fails before 0132 or holds both ways)."""

    rt, out = h.reply_text, []
    tools = TOOLS
    if not hasattr(rt.GlmCallStreamer, "whole"):
        return [("streamer: pieces before the closer", ["the streamer sends each call whole (before 0132)"])]
    # 1. a spelled tag the mask made inert, split at every point between two pieces: given back whole, never U+E000
    marks = h.app.reply_ids
    inert = marks.inert["</arg_value>"]
    text = ("<tool_call>Bash<arg_key>command</arg_key><arg_value>echo '" + inert + "' && echo '" +
            marks.inert["<tool_call>"] + "'</arg_value></tool_call>")
    problems = []
    for step in range(1, 9):
        st = rt.GlmCallStreamer(tools, unmask=marks.unmask)
        deltas = []
        for n in range(step, len(text) + step, step):
            deltas += st.feed(text[:n])
        joined = "".join(c["function"].get("arguments", "") for d in deltas for c in d["tool_calls"])
        want = J({"command": "echo '</arg_value>' && echo '<tool_call>'"})
        if joined != want or chr(0xE000) in joined:
            problems.append(f"step {step}: {joined!r}")
    out.append(("streamer: an inert tag split between pieces comes back whole", problems))
    # 2. a text that does not extend the one read: the streamer stops for the reply, nothing more goes out
    st = rt.GlmCallStreamer(tools)
    first = st.feed("<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la")
    later = st.feed("<tool_call>Bash<arg_key>command</arg_key><arg_value>rm -rf x</arg_value></tool_call>")
    problems = [] if st.off and first and not later and not st.whole() else [f"off={st.off} later={later}"]
    out.append(("streamer: a text that changed under it turns it off", problems))
    # 3. pieces before the closer: a string value streams as written, its brace only at </tool_call>
    st = rt.GlmCallStreamer(tools)
    a = st.feed("<tool_call>Write<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value><arg_key>content</arg_key>"
                "<arg_value>line one\n")
    b = st.feed("<tool_call>Write<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value><arg_key>content</arg_key>"
                "<arg_value>line one\nline two</arg_value>")
    c = st.feed("<tool_call>Write<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value><arg_key>content</arg_key>"
                "<arg_value>line one\nline two</arg_value></tool_call>")
    args = lambda ds: "".join(x["function"].get("arguments", "") for d in ds for x in d["tool_calls"])  # noqa: E731
    problems = []
    if args(a) != '{"file_path":"/tmp/a","content":"line one\\n' or args(b) != 'line two"' or args(c) != "}":
        problems.append(f"pieces {args(a)!r} {args(b)!r} {args(c)!r}")
    out.append(("streamer: a value streams, its brace waits for </tool_call>", problems))
    # 4. a partial closer at a piece's end waits; a tail that only looks like one goes out with the next piece
    st = rt.GlmCallStreamer(tools)
    head = "<tool_call>Bash<arg_key>command</arg_key><arg_value>a </"
    a = st.feed(head)
    b = st.feed(head + "b")
    problems = [] if args(a) == '{"command":"a ' and args(b) == "</b" else [f"{args(a)!r} {args(b)!r}"]
    out.append(("streamer: a tail that may begin a closer waits", problems))
    return out



FUZZ_TOOLS = TOOLS + [fn("set_flags", {"enabled": {"type": "boolean"}, "level": {"type": "integer"},
                                                       "ratio": {"type": "number"}, "tags": {"type": "array"},
                                                       "meta": {"type": "object"}, "mode": {"enum": ["a", "b"]},
                                                       "note": {"type": "string"}})]
SCHEMA_KIND = {}
for t in FUZZ_TOOLS:
    for k, s in t["function"]["parameters"]["properties"].items():
        SCHEMA_KIND[(t["function"]["name"], k)] = s

WORDS = ["the", "file", "rate", "token", "bucket", "src/", "def", "return", "if", "else", "True", "None", "{", "}",
         "[", "]", "(", ")", ":", ",", "=", "==", "->", "self", "import", "json", "value", "key", "path", "line",
         "速率", "限制", "令牌", "🚀", "👩‍💻", "café", "naïve", "Ελλάδα", "русский", "日本語", "—", "…", "→", "\t",
         "\\n", "\\", "\\\\", '"', "'", "`", "\n", "\n\n", "  ", "\r\n", "\x01", "\x7f", " ", "<", ">", "</",
         "<arg", "arg_value", "<tool", "null", "0", "42", "-7", "3.14", "1e9", "%s", "{}", "[]", "�"]
FUZZ_SPELLED = ["<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</think>",
           "<think>"]


SPELLED_SEEN = [False]       # whether the reply being made spells markup in its text (real markup to the unit test)
SPELL_RATE = [0.03]          # a word's chance to be spelled markup


def rtext(rng: random.Random, lo: int = 0, hi: int = 40, spelled: float | None = None) -> str:
    spelled = SPELL_RATE[0] if spelled is None else spelled
    out = []
    for _ in range(rng.randint(lo, hi)):
        if rng.random() < spelled:
            SPELLED_SEEN[0] = True
            out.append(rng.choice(FUZZ_SPELLED))
        else:
            out.append(rng.choice(WORDS))
        if rng.random() < 0.5:
            out.append(" ")
    return "".join(out)


def json_text(rng: random.Random, kind: str) -> str:
    value: Any
    if kind == "boolean":
        value = rng.choice([True, False])
    elif kind == "integer":
        value = rng.randint(-1000, 10 ** 6)
    elif kind == "number":
        value = rng.choice([rng.randint(-50, 50), round(rng.uniform(-1e3, 1e3), 3)])
    elif kind == "array":
        value = [rtext(rng, 0, 3, 0) for _ in range(rng.randint(0, 3))]
    elif kind == "object":
        value = {rtext(rng, 1, 2, 0): rng.choice([1, "x", None, [1, 2]]) for _ in range(rng.randint(0, 3))}
    else:
        value = rtext(rng, 0, 4, 0)
    text = json.dumps(value, ensure_ascii=rng.random() < 0.2, indent=rng.choice([None, None, 2]))
    r = rng.random()
    if r < 0.06:
        return text[:-1] if text[-1:] in "]}" else text           # #87: one closer short
    if r < 0.10:
        return rtext(rng, 1, 3, 0)                                 # not JSON: stays text
    if r < 0.14 and kind in ("integer", "number"):
        return '"' + str(value) + '"'                              # quoted: stays text
    return (" " if rng.random() < 0.1 else "") + text + ("\n" if rng.random() < 0.1 else "")


def value_for(rng: random.Random, tool: str, key: str) -> str:
    schema = SCHEMA_KIND.get((tool, key), {})
    kind = schema.get("type")
    if isinstance(kind, list):
        if "null" in kind and rng.random() < 0.5:
            return "null"
        if "object" in kind and rng.random() < 0.5:
            return json_text(rng, "object")
        return rtext(rng, 0, 30)
    if kind in ("boolean", "integer", "number", "array", "object"):
        return json_text(rng, kind)
    if "enum" in schema:
        return rng.choice(schema["enum"] + ["c"])
    if schema == {} and rng.random() < 0.3:
        return json_text(rng, "object")
    if rng.random() < 0.15:
        return rtext(rng, 60, 300, 0.01)                          # a long value (a file)
    return rtext(rng, 0, 30)


class Reply:
    """A reply as parts (strings, marker tuples), and what kind it is."""

    def __init__(self) -> None:
        self.parts: list[Any] = []
        self.odd = False            # markup the streamer may not follow as the parser reads it
        self.repeated = False
        self.long_values: list[int] = []      # lengths of long string values, by call order
        self.spelled = False        # its text spells markup (words with patch 0117, markup to the bare streamer)
        self.foreign = False        # a JSON or Qwen call among GLM's (the stream sends those after GLM's)


def make_call(rng: random.Random, reply: Reply, odd_rate: float) -> None:
    tool = rng.choice(FUZZ_TOOLS)
    name = tool["function"]["name"]
    if rng.random() < 0.04:
        name = rng.choice(["spawn_agent", "write", "functions.exec"])   # unoffered (or another spelling)
    keys = list(tool["function"]["parameters"]["properties"])
    nl = "\n" if rng.random() < 0.2 else ""
    if rng.random() < 0.03:                                     # a JSON or Qwen call between GLM's
        reply.foreign = True
        reply.parts += [M("<tool_call>"), json.dumps({"name": name, "arguments": {keys[0]: "x"}}) if rng.random() < 0.5
                        else f"<function={name}><parameter={keys[0]}>x</parameter></function>", M("</tool_call>")]
        return
    reply.parts += [M("<tool_call>"), (" " if rng.random() < 0.05 else "") + name + nl]
    chosen = rng.sample(keys, rng.randint(0, len(keys)))
    if rng.random() < 0.1:
        chosen.append(rng.choice(["verbose", "extra_arg", "k"]))   # unknown key
    if chosen and rng.random() < 0.05:
        chosen.append(rng.choice(chosen))                          # repeated key
        reply.repeated = True
    longest = 0
    for k in chosen:
        v = value_for(rng, name, k)
        is_text = not (SCHEMA_KIND.get((name, k), {}).get("type") in ("boolean", "integer", "number", "array",
                                                                         "object"))
        if is_text:
            longest = max(longest, len(v))
        odd = rng.random() < odd_rate
        if odd:
            reply.odd = True
            kind = rng.randrange(6)
            if kind == 0:       # an <arg_key> left out (#22)
                reply.parts += [nl or "\n", k, M("</arg_key>"), M("<arg_value>"), v, M("</arg_value>")]
            elif kind == 1:     # text after </arg_key>
                reply.parts += [M("<arg_key>"), k, M("</arg_key>"), rtext(rng, 1, 3, 0), M("<arg_value>"), v,
                                M("</arg_value>")]
            elif kind == 2:     # the block's closer inside the value
                reply.parts += [M("<arg_key>"), k, M("</arg_key>"), M("<arg_value>"), v[:len(v) // 2],
                                M("</tool_call>"), v[len(v) // 2:], M("</arg_value>")]
            elif kind == 3:     # a value never closed
                reply.parts += [M("<arg_key>"), k, M("</arg_key>"), M("<arg_value>"), v]
            elif kind == 4:     # markup tokens inside a value
                reply.parts += [M("<arg_key>"), k, M("</arg_key>"), M("<arg_value>"), v,
                                M(rng.choice(["<arg_key>", "</arg_key>", "<arg_value>", "<tool_call>"])), v,
                                M("</arg_value>")]
            else:               # a key never closed
                reply.parts += [M("<arg_key>"), k, M("<arg_value>"), v, M("</arg_value>")]
        else:
            reply.parts += [M("<arg_key>"), k, M("</arg_key>")] + ([nl] if nl else []) + \
                           [M("<arg_value>"), v, M("</arg_value>")] + ([nl] if nl else [])
    if longest >= 120:
        reply.long_values.append(longest)
    reply.parts.append(M("</tool_call>"))


def make_reply(rng: random.Random, thinking: bool, odd_rate: float) -> Reply:
    SPELLED_SEEN[0] = False
    reply = Reply()
    if thinking:
        reply.parts.append(rtext(rng, 1, 30))
        if rng.random() < 0.05:                       # a call in the think block
            make_call(rng, reply, odd_rate)
            reply.parts.append(rtext(rng, 0, 5))
        reply.parts.append(M("</think>"))
        if rng.random() < 0.5:
            reply.parts.append("\n")
    if rng.random() < 0.3:
        reply.parts.append(rtext(rng, 1, 20))
    for i in range(rng.choice([0, 1, 1, 1, 2, 2, 3])):
        make_call(rng, reply, odd_rate)
        if rng.random() < 0.2:
            reply.parts.append(rng.choice(["\n", " ", rtext(rng, 1, 8)]))
    if rng.random() < 0.15:
        reply.parts.append(rtext(rng, 1, 10))
    if rng.random() < 0.95:
        reply.parts.append(M(rng.choice([EOS, END, "<|endoftext|>"])))
    reply.spelled = SPELLED_SEEN[0]
    return reply


# -- unit: the streamer against the parser ---------------------------------------------------------------------------
MARK_TEXT = {m: m for m in MARKS}
END_TEXT = {EOS, END, "<|endoftext|>"}


def unit(rt: Any, n: int, seed: int) -> dict[str, Any]:
    """``GlmCallStreamer`` (``rt``: tensorfold.cuda.reply_text) against ``parse_tool_calls`` on ``n`` random replies."""

    stats = {"replies": 0, "calls_parsed": 0, "closed": 0, "left_open": 0, "open_at_cut": 0, "well_formed": 0,
             "odd": 0, "repeated": 0, "fail": 0, "streaming_fail": 0, "long_values": 0, "single_delta_long": 0}
    SPELL_RATE[0] = 0.004       # spelled markup is real markup to the bare streamer: an odd reply here
    examples: list[str] = []
    for i in range(n):
        rng = random.Random(seed * 1_000_003 + i)
        reply = make_reply(rng, thinking=False, odd_rate=rng.choice([0.0, 0.0, 0.15]))
        text = "".join(p[0] if isinstance(p, tuple) else p for p in reply.parts if not (isinstance(p, tuple) and
                                                                                         p[0] in END_TEXT))
        full = len(text)
        cut = full if rng.random() < 0.8 else rng.randint(0, full)
        text = text[:cut]
        whole_text = cut == full and not reply.odd and not reply.spelled     # spelled markup is real markup here
        stream = rt.GlmCallStreamer(FUZZ_TOOLS)
        deltas: list[dict[str, Any]] = []
        at = 0
        with contextlib.redirect_stdout(io.StringIO()):
            while at < len(text):
                at = min(len(text), at + rng.choice([1, 1, 2, 3, 5, 8, 13, 40]))
                deltas += stream.feed(text[:at])
            deltas += stream.feed(text)
        parsed = rt.parse_tool_calls(text, FUZZ_TOOLS)[1] or []
        got: dict[int, dict[str, Any]] = {}
        for d in deltas:
            for c in d["tool_calls"]:
                slot = got.setdefault(c["index"], {"name": "", "args": "", "pieces": 0, "head": 0})
                if "id" in c:
                    slot["head"] += 1
                    slot["name"] = c["function"]["name"]
                if c["function"].get("arguments"):
                    slot["args"] += c["function"]["arguments"]
                    slot["pieces"] += 1
        problems = []
        if sorted(got) != list(range(len(got))) or any(s["head"] != 1 for s in got.values()):
            problems.append("headers")
        stock = not hasattr(stream, "whole")             # before 0132: each call whole at its closer
        whole_calls = stream.calls[:stream.sent] if stock else stream.whole()
        status = ["whole"] * stream.sent if stock else stream.status
        for idx, call in enumerate(stream.calls):
            if (not stock or idx < stream.sent) and got.get(idx, {}).get("args") != call["function"]["arguments"]:
                problems.append(f"call {idx}: the pieces do not join to what it holds")
        closed = [(c["function"]["name"], c["function"]["arguments"]) for c in whole_calls]
        allp = [(c["function"]["name"], c["function"]["arguments"]) for c in parsed]
        rest = [(c["function"]["name"], c["function"]["arguments"]) for c in stream.rest(FUZZ_TOOLS)]
        # every closed call is the parser's, in order (a subsequence of the parser's calls)
        it = iter(allp)
        if reply.repeated:
            pj = [json.dumps(json.loads(a)) for _, a in allp]
            it2 = iter(pj)
            if not all(any(json.dumps(json.loads(a)) == q for q in it2) for _, a in closed):
                problems.append(f"closed calls (repeated key) {closed} are not the parser's {allp}")
        elif not all(any(c == p for p in it) for c in closed):
            problems.append(f"closed calls {closed} are not the parser's {allp}")
        if whole_text:
            stats["well_formed"] += 1
            if not reply.repeated and sorted(closed + rest) != sorted(allp):
                problems.append(f"well formed: closed {closed} + skipped {rest} != parser's {allp}")
            if stream.open or any(s == "left" for s in status):
                problems.append("well formed: a call left open")
        else:
            stats["odd"] += 1
        stats["repeated"] += reply.repeated
        stats["replies"] += 1
        stats["calls_parsed"] += len(allp)
        stats["closed"] += len(closed)
        stats["left_open"] += sum(1 for s in status if s == "left")     # the parser read its block otherwise
        stats["open_at_cut"] += bool(stream.open)                               # the reply ended inside it
        streaming_problem = False
        if whole_text and not reply.repeated:
            for (name, args), slot in zip(closed, [got[i] for i, s in enumerate(status) if s == "whole"]):
                if len(args) >= 200 and any(len(v) >= 120 for v in json.loads(args).values() if isinstance(v, str)):
                    stats["long_values"] += 1
                    if slot["pieces"] < 2:
                        stats["single_delta_long"] += 1
                        streaming_problem = True
        if problems:
            stats["fail"] += 1
            stats["fail_well_formed"] = stats.get("fail_well_formed", 0) + whole_text
            if len(examples) < 5:
                examples.append(f"reply {i} (seed {seed}): {problems[:2]}\n    text: {text[:400]!r}")
        stats["streaming_fail"] += streaming_problem
    stats["examples"] = examples  # type: ignore[assignment]
    return stats




# -- main -------------------------------------------------------------------------------------------------------------
# sha256 (first 16 hex digits) of each reply 0132 must not change, as the engine before it wrote them: every whole
# reply, and every stream of a case that sends no call while it decodes. Recorded with --print-golden on main's patches
# (0001-0129) and the checkpoint Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (its tokenizer and template).
GOLDEN: dict[str, str] = {
    "call_in_think_block/chat-stream": "7a04ec822e3ff6d2",
    "call_in_think_block/chat-whole": "1c7f56fd4c2058be",
    "call_in_think_block/messages-stream": "cc2a7af48f32883e",
    "call_in_think_block/messages-whole": "63a6ff4e04fac47a",
    "cut_by_max_tokens/chat-whole": "e07494254cc9f11f",
    "cut_by_max_tokens/messages-whole": "4a4d7f0be39a98e7",
    "edit/chat-whole": "03b64211233aba83",
    "edit/messages-whole": "d1a3ec4e48d06789",
    "end_before_closer/chat-whole": "5b773fcd78316d29",
    "end_before_closer/messages-whole": "404b7ec4da3c1926",
    "long_call_in_think_block/chat-whole": "2ca00ca07bce6ee9",
    "long_call_in_think_block/messages-whole": "11a2816a1c059688",
    "missing_arg_key/chat-whole": "753f4285275f01d2",
    "missing_arg_key/messages-whole": "56e25d36a8981417",
    "newline_style/chat-whole": "57b8ecbc702dcc83",
    "newline_style/messages-whole": "3f251c4f07f7e393",
    "no_tools/chat-stream": "e66be9378e765074",
    "no_tools/chat-stream-smooth": "fbba54369b5f17c4",
    "no_tools/chat-whole": "90ccd52b85583d74",
    "no_tools/messages-stream": "77a9aae4bfb09300",
    "no_tools/messages-whole": "c9496f1fd369e446",
    "no_tools_thinking_off/chat-stream": "793782e49428f7f3",
    "no_tools_thinking_off/chat-whole": "ebfb1e6e3a941689",
    "no_tools_thinking_off/messages-stream": "1b7425eaa36455a8",
    "no_tools_thinking_off/messages-whole": "0695d1b35b48521c",
    "one_call_policy/chat-stream": "cc5148b934d48872",
    "one_call_policy/chat-whole": "c26695fce71b9aee",
    "one_call_policy/messages-stream": "08929fff7067c588",
    "one_call_policy/messages-whole": "45bc9602f0e88009",
    "parallel_two/chat-whole": "c0c5c108caa7eee1",
    "parallel_two/messages-whole": "e3d1af06ef747aba",
    "prose_after_call/chat-whole": "19c4d65b86d1b966",
    "prose_after_call/messages-whole": "3dfc6f21264ce47b",
    "prose_then_call/chat-whole": "bff7b4dd22bb5744",
    "prose_then_call/messages-whole": "8ca8ab149e11b7f4",
    "quoted_calls/chat-whole": "fe4fc4ad05191f67",
    "quoted_calls/messages-whole": "9f88a1a1edc744f2",
    "repeated_key/chat-whole": "0a5b41cf27791d27",
    "repeated_key/messages-whole": "a198cbf130629801",
    "spelled_closers/chat-whole": "063261f35898361b",
    "spelled_closers/messages-whole": "e854620edb5b0dc4",
    "stop_in_value/chat-whole": "6a466d608d5b3f36",
    "stop_in_value/messages-whole": "68333baa4a986c62",
    "tools_no_call/chat-stream": "04d52f9dad6a200d",
    "tools_no_call/chat-whole": "7c6825e592c6a157",
    "tools_no_call/messages-stream": "b094b92ec9eaf979",
    "tools_no_call/messages-whole": "c97e615e29d4290a",
    "typed_values/chat-whole": "06b68b363032f885",
    "typed_values/messages-whole": "c759384de31f154a",
    "unicode_write/chat-whole": "43d721fbb91b597f",
    "unicode_write/messages-whole": "54021593ed9da1b8",
    "unoffered_tool/chat-whole": "2b5a9ee15a366c5f",
    "unoffered_tool/messages-whole": "6b3e9e45df57a319",
    "write_long/chat-whole": "a8c34c59cfe7541b",
    "write_long/messages-whole": "afec8b099228b240",
}


def digest(text: str) -> str:
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help="the checkpoint's directory (tokenizer.json, chat_template.jinja)")
    ap.add_argument("--source-root", help="TensorFold src/ with the patches applied (default: the installed one)")
    ap.add_argument("--expect-stock", action="store_true", help="the source has no 0132: the old stream must show")
    ap.add_argument("--fuzz", type=int, default=20000, help="random replies for the streamer (0: none)")
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--only", help="one case")
    ap.add_argument("--print-golden", action="store_true", help="print GOLDEN, the hashes of this source's replies")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    h = Harness(a.source_root, a.checkpoint)
    glm = Glm(h.tok)
    fails = stock_like = total = 0
    golden: dict[str, str] = {}
    try:
        for case in CASES:
            if a.only and case.name != a.only:
                continue
            total += 1
            raw = run_case(h, case, glm.ids(case.parts))
            kept = sorted(lbl for lbl in raw if lbl.endswith("whole") or case.name in UNCHANGED_STREAMS)
            if a.print_golden:
                golden.update({f"{case.name}/{lbl}": digest(raw[lbl]) for lbl in kept})
                continue
            problems, streaming = check(case, raw)
            for lbl in kept:
                got, want = digest(raw[lbl]), GOLDEN.get(f"{case.name}/{lbl}")
                if got != want:
                    problems.append(f"{lbl}: not the reply of the engine before 0132 (hash {got}, want {want})")
            if a.expect_stock:      # the old stream shows (the long calls' at least), and nothing else differs
                ok = not problems and (bool(streaming) or not case.write_like)
                stock_like += bool(streaming)
            else:
                ok = not problems and not streaming
            fails += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {case.name}" + (f"  ({case.note})" if case.note and a.v else ""))
            for p in problems + ([] if a.expect_stock and ok else streaming):
                print(f"     {p}")
            if a.expect_stock and streaming and a.v:
                for p in streaming:
                    print(f"     before 0132: {p}")
        if not a.print_golden and not a.only:
            for name, problems in streamer_checks(h):
                total += 1
                ok = bool(problems) if a.expect_stock else not problems    # before 0132: each call whole
                fails += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {name}")
                for p in ([] if a.expect_stock and ok else problems):
                    print(f"     {p}")
            if a.fuzz > 0:
                total += 1
                with contextlib.redirect_stdout(io.StringIO()):
                    st = unit(h.reply_text, a.fuzz, a.seed)
                examples = st.pop("examples")
                # before 0132 the long values arrive in one delta; the closed-call checks are reported, not judged
                ok = st["single_delta_long"] > 0 if a.expect_stock else st["fail"] == 0 and st["streaming_fail"] == 0
                fails += not ok
                print(f"{'ok  ' if ok else 'FAIL'} fuzz: {st['replies']} random replies (seed {a.seed}, "
                      f"{st['well_formed']} well formed): {st['fail']} with a problem (a closed call the parser does "
                      f"not return, or pieces that do not join), {st['single_delta_long']} of {st['long_values']} long "
                      f"values in one delta, {st['left_open']} calls left open, {st['open_at_cut']} cut inside a call")
                for e in (examples if not ok or a.v else []):
                    print(f"     {e}")
    finally:
        h.close()
    if a.print_golden:
        print("GOLDEN: dict[str, str] = {")
        for key in sorted(golden):
            print(f'    "{key}": "{golden[key]}",')
        print("}")
        return 0
    if a.expect_stock:
        print(f"{total - fails}/{total} as expected before 0132: {stock_like} cases stream otherwise (each long call "
              f"arrives whole), every whole reply and every stream without a streamed call has its recorded hash")
    else:
        print(f"{total - fails}/{total} ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
