"""CPU-only test of patch 0113: a call to a tool the request did not offer goes out under its own name (#103).

  python3 -B tools/test_unoffered_tool_call.py --source-root /path/to/patched/src
No Torch/CUDA, sockets or source writes. Runs replies through the real ``ThinkSplit``, ``hide_tool_calls``,
``GlmCallStreamer`` and ``parse_tool_calls`` as the CUDA server does (whole, streamed one character at a time, and the
one-call mode) and checks the content and the calls. Before 0113 such a call stayed content text (finish "stop", no
tool_calls), so the client never learned the tool is unknown and the agent stopped. The rule is TensorFold's
be8bccbec (#256/#415): the name goes out when it is 1 to 64 of [A-Za-z0-9_-] and the arguments a finite JSON object;
anything else stays text, and bare JSON outside an envelope still needs an offered name."""
from __future__ import annotations

import argparse
import json
import sys

TOOLS = [{"type": "function", "function": {"name": "codemode", "parameters": {
    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
PASEO = ("<tool_call>mcp__paseo__create_agent<arg_key>provider</arg_key><arg_value>pi/dgx-spark/GLM-5.3-Flash-EXL3"
         "</arg_value><arg_key>title</arg_key><arg_value>hi</arg_value></tool_call>")
OFFERED = "<tool_call>CodeMode<arg_key>code</arg_key><arg_value>1 + 1</arg_value></tool_call>"
PASEO_ARGS = {"provider": "pi/dgx-spark/GLM-5.3-Flash-EXL3", "title": "hi"}

# (name, reply after </think>, content, [(name, arguments)], one-call mode's [(name, arguments)])
CASES = [
    ("unoffered GLM call (the paseo case)", PASEO, "", [("mcp__paseo__create_agent", PASEO_ARGS)],
     [("mcp__paseo__create_agent", PASEO_ARGS)]),
    ("prose, then an unoffered call", "Creating it directly." + PASEO, "Creating it directly.",
     [("mcp__paseo__create_agent", PASEO_ARGS)], [("mcp__paseo__create_agent", PASEO_ARGS)]),
    ("offered keeps its spelling, then unoffered", OFFERED + PASEO, "",
     [("codemode", {"code": "1 + 1"}), ("mcp__paseo__create_agent", PASEO_ARGS)], [("codemode", {"code": "1 + 1"})]),
    ("unoffered JSON call", '<tool_call>{"name": "launch", "arguments": {"when": "now"}}</tool_call>', "",
     [("launch", {"when": "now"})], [("launch", {"when": "now"})]),
    ("unoffered Qwen call", "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>", "",
     [("launch", {"when": "now"})], [("launch", {"when": "now"})]),
    # stay text, as upstream
    ("name not a function name", '<tool_call>{"name": "launch rocket!", "arguments": {"when": "now"}}</tool_call>',
     '<tool_call>{"name": "launch rocket!", "arguments": {"when": "now"}}</tool_call>', [], []),
    ("name with a dot", "<tool_call>functions.exec<arg_key>input</arg_key><arg_value>1</arg_value></tool_call>",
     "<tool_call>functions.exec<arg_key>input</arg_key><arg_value>1</arg_value></tool_call>", [], []),
    ("name past 64 characters", "<tool_call>" + "a" * 65 + "<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>",
     "<tool_call>" + "a" * 65 + "<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>", [], []),
    ("arguments not finite", '<tool_call>{"name": "launch", "arguments": {"when": NaN}}</tool_call>',
     '<tool_call>{"name": "launch", "arguments": {"when": NaN}}</tool_call>', [], []),
    ("bare JSON naming an unoffered tool", '{"name": "launch", "arguments": {}}', '{"name": "launch", "arguments": {}}',
     [], []),
]


def run(mod, answer: str, streamed: bool, max_calls: int | None) -> tuple[str, list[tuple[str, dict]]]:
    reply = "Plan.</think>" + answer
    think, shown = mod.ThinkSplit(mod.THINK_CALL_HOLD if streamed else None), ""
    streamer = mod.GlmCallStreamer(TOOLS) if streamed and max_calls is None else None
    for n in range(1, len(reply)) if streamed else ():
        _, ans = think(reply[:n], False)
        content = mod.hide_tool_calls(ans, finished=False)
        assert content.startswith(shown), f"content rewrote {shown[-40:]!r} as {content[-40:]!r}"
        shown = content
        if streamer is not None and think.stream_from is not None:
            streamer.feed(reply[:n][think.stream_from:])
    _, ans = think(reply, True)
    content, calls = mod.parse_tool_calls(ans, TOOLS, max_calls=max_calls)
    if streamer is not None:                       # as server.py: the streamer's calls, then the blocks it skipped
        streamer.feed(reply[think.stream_from:])
        streamed_calls = [*streamer.calls[:streamer.sent], *streamer.rest(TOOLS)]
        assert sorted(c["function"]["arguments"] for c in streamed_calls) == \
               sorted(c["function"]["arguments"] for c in calls or ()), "streamed calls differ from the end parser's"
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or ()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="patched TensorFold src/ (holds tensorfold/)")
    sys.path.insert(0, ap.parse_args().source_root)
    import tensorfold.cuda.reply_text as mod

    failed = 0
    for name, answer, want_content, want_calls, want_single in CASES:
        for label, streamed, max_calls, want in (("whole", False, None, want_calls), ("streamed", True, None, want_calls),
                                                 ("one call", False, 1, want_single)):
            try:
                content, calls = run(mod, answer, streamed, max_calls)
                problems = [] if calls == want else [f"calls {calls}"]
                if max_calls is None and content.strip() != want_content.strip():
                    problems.append(f"content {content[:70]!r}")
            except AssertionError as exc:
                problems = [str(exc)]
            print(f"{'ok  ' if not problems else 'FAIL'} {name} [{label}]" + ("" if not problems else
                                                                            ": " + "; ".join(problems)))
            failed += bool(problems)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
