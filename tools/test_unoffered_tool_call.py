"""CPU-only test of patch 0115: a call to a tool the request did not offer stays text and is reported (#103).

  python3 -B tools/test_unoffered_tool_call.py --source-root /path/to/patched/src
No Torch/CUDA, sockets or source writes. Runs replies through the real ``ThinkSplit``, ``hide_tool_calls``,
``GlmCallStreamer`` and ``parse_tool_calls`` as the CUDA server does (whole, streamed one character at a time, and the
one-call mode) and checks the content, the calls and the rejected names; then the server's ``tool_call_rejections``
(its log line, and ``tool_call_rejections`` in the stats only with TF_GLM_TOOL_DIAGNOSTICS=1). The rule is TensorFold's
since 7be90a36b / 806af974c (v1.0.0 on): only an offered name is a call. Before 0115 such a call stayed text silently."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

TOOLS = [{"type": "function", "function": {"name": "codemode", "parameters": {
    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
PASEO = ("<tool_call>mcp__paseo__create_agent<arg_key>provider</arg_key><arg_value>pi/dgx-spark/GLM-5.3-Flash-EXL3"
         "</arg_value><arg_key>title</arg_key><arg_value>hi</arg_value></tool_call>")
OFFERED = "<tool_call>CodeMode<arg_key>code</arg_key><arg_value>1 + 1</arg_value></tool_call>"
JSON_CALL = '<tool_call>{"name": "launch", "arguments": {"when": "now"}}</tool_call>'
QWEN_CALL = "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>"
DOTTED = "<tool_call>functions.exec<arg_key>input</arg_key><arg_value>1</arg_value></tool_call>"
CODEMODE = [("codemode", {"code": "1 + 1"})]

# (name, reply after </think>, content, [(name, arguments)], one-call mode's [(name, arguments)], rejected names)
CASES = [
    # an undeclared call stays text, and its name is reported
    ("unoffered GLM call (the paseo case)", PASEO, PASEO, [], [], ["mcp__paseo__create_agent"]),
    ("prose, then an unoffered call", "Creating it directly." + PASEO, "Creating it directly." + PASEO, [], [],
     ["mcp__paseo__create_agent"]),
    ("unoffered call without arguments", "<tool_call>list_agents</tool_call>", "<tool_call>list_agents</tool_call>",
     [], [], ["list_agents"]),
    ("unoffered JSON call", JSON_CALL, JSON_CALL, [], [], ["launch"]),
    ("unoffered Qwen call", QWEN_CALL, QWEN_CALL, [], [], ["launch"]),
    ("unoffered dotted name", DOTTED, DOTTED, [], [], ["functions.exec"]),
    # an offered call is still a call, and keeps its offered spelling
    ("offered call", OFFERED, "", CODEMODE, CODEMODE, []),
    ("offered, then unoffered", OFFERED + PASEO, PASEO, CODEMODE, CODEMODE, ["mcp__paseo__create_agent"]),
    # prose naming the tags is no call and no rejection (0112)
    ("tags around a word", "The tags are <tool_call> and </tool_call>.", "The tags are <tool_call> and </tool_call>.",
     [], [], []),
    ("tags around a placeholder", "Wrap each call as <tool_call> name </tool_call> in the reply.",
     "Wrap each call as <tool_call> name </tool_call> in the reply.", [], [], []),
    ("bare JSON naming an unoffered tool", '{"name": "launch", "arguments": {}}', '{"name": "launch", "arguments": {}}',
     [], [], []),
]


def run(mod, answer: str, streamed: bool, max_calls: int | None) -> tuple[str, list[tuple[str, dict]], list[str]]:
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
    rejected: list[str] = []
    content, calls = mod.parse_tool_calls(ans, TOOLS, max_calls=max_calls, rejected=rejected)
    if streamer is not None:                       # as server.py: the streamer's calls, then the blocks it skipped
        streamer.feed(reply[think.stream_from:])
        streamed_calls = [*streamer.calls[:streamer.sent], *streamer.rest(TOOLS)]
        assert [(c["function"]["name"], c["function"]["arguments"]) for c in streamed_calls] == \
               [(c["function"]["name"], c["function"]["arguments"]) for c in calls or ()], \
               "streamed calls differ from the end parser's"
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or ()], rejected


def check_report(server) -> list[str]:
    """The server's report of rejected names: a log line always, the stats field only with TF_GLM_TOOL_DIAGNOSTICS=1."""

    problems, stats = [], {"token_sha": "x"}
    for mode, want in ((None, stats), ("0", stats), ("1", {**stats, "tool_call_rejections": [
            {"reason": "unknown_tool_name", "name": "exec"}]})):
        os.environ.pop("TF_GLM_TOOL_DIAGNOSTICS", None)
        if mode is not None:
            os.environ["TF_GLM_TOOL_DIAGNOSTICS"] = mode
        with contextlib.redirect_stdout(io.StringIO()) as out:
            got = server.tool_call_rejections(["exec"], 2, stats)
        if got != want:
            problems.append(f"TF_GLM_TOOL_DIAGNOSTICS={mode}: stats {got}")
        if "tool call rejected: unknown_tool_name ['exec'] (2 tools offered); kept as text" not in out.getvalue():
            problems.append(f"TF_GLM_TOOL_DIAGNOSTICS={mode}: log {out.getvalue()!r}")
    os.environ.pop("TF_GLM_TOOL_DIAGNOSTICS", None)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="patched TensorFold src/ (holds tensorfold/)")
    sys.path.insert(0, ap.parse_args().source_root)
    import tensorfold.cuda.reply_text as mod
    import tensorfold.cuda.server as server

    failed = 0
    for name, answer, want_content, want_calls, want_single, want_rejected in CASES:
        for label, streamed, max_calls, want in (("whole", False, None, want_calls), ("streamed", True, None, want_calls),
                                                 ("one call", False, 1, want_single)):
            try:
                content, calls, rejected = run(mod, answer, streamed, max_calls)
                problems = [] if calls == want else [f"calls {calls}"]
                if max_calls is None and content.strip() != want_content.strip():
                    problems.append(f"content {content[:70]!r}")
                if max_calls is None and rejected != want_rejected:
                    problems.append(f"rejected {rejected}")
            except AssertionError as exc:
                problems = [str(exc)]
            print(f"{'ok  ' if not problems else 'FAIL'} {name} [{label}]" + ("" if not problems else
                                                                            ": " + "; ".join(problems)))
            failed += bool(problems)
    problems = check_report(server)
    print(f"{'ok  ' if not problems else 'FAIL'} server report" + ("" if not problems else ": " + "; ".join(problems)))
    failed += bool(problems)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
