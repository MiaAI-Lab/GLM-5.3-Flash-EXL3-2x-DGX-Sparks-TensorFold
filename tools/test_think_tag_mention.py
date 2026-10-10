"""CPU-only test of patch 0112: a ``<tool_call>`` tag a reply mentions is text, not the start of a call.

  python3 -B tools/test_think_tag_mention.py --source-root /path/to/patched/src
No Torch/CUDA, sockets or source writes. Runs replies through the real ``ThinkSplit``, ``parse_tool_calls``,
``hide_tool_calls`` and ``GlmCallStreamer`` as the CUDA server does, whole and streamed one character at a time (with
and without the call hold), and checks the reasoning, the content and the calls. Before 0112 a mention matched up to
the next ``</tool_call>``: in the think block through ``</think>`` to the real call, in the answer through the real
call, so the call came back as content text with no tool_calls (live on v1.10: every time the model quoted the tag)."""
from __future__ import annotations

import argparse
import json
import sys

BASH = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def call(command: str) -> str:
    return f"<tool_call>bash<arg_key>command</arg_key><arg_value>{command}</arg_value></tool_call>"


CALL, GREP = call("grep -c x s.jsonl"), call("grep -c '<tool_call>' s.jsonl")      # GREP's value holds the tag
LONG = " More reasoning about the session file and what the grep should count." * 20     # > THINK_CALL_HOLD
# the reply seen live on v1.10 (both blocks shortened): the tag in the reasoning and the answer, and in the value
LIVE = ("The user says session.jsonl contains a literal tag `<tool_call>` followed by `<arg_key>` markup. I'll run it."
        "</think>The file contains the literal markup `<tool_call>` instead of an actual tool call. I'll check the saved "
        "session file for that literal string." + call("grep -c '<tool_call>' /tmp/workspace/documents/session.jsonl"))

# (name, reply, reasoning, content, calls' commands); reasoning and content compared stripped
CASES = [
    # the failures: the tag mentioned, then a real call
    ("mention, close, call", "The literal `<tool_call>...` markup. Let me grep.</think>" + CALL,
     "The literal `<tool_call>...` markup. Let me grep.", "", ["grep -c x s.jsonl"]),
    ("mention twice, close, call", "Count `<tool_call>` and `<arg_key>`; `<tool_call>` again.</think>\n" + CALL,
     "Count `<tool_call>` and `<arg_key>`; `<tool_call>` again.", "", ["grep -c x s.jsonl"]),
    ("mention, long reasoning, close, call", "The tag `<tool_call>` leaked." + LONG + "</think>" + CALL,
     "The tag `<tool_call>` leaked." + LONG, "", ["grep -c x s.jsonl"]),
    ("name-like mention", "A <tool_call>bash and then prose.</think>" + CALL, "A <tool_call>bash and then prose.", "",
     ["grep -c x s.jsonl"]),
    ("mention in the answer, call", "Plan.</think>It has the literal `<tool_call>` markup. I'll count it." + GREP,
     "Plan.", "It has the literal `<tool_call>` markup. I'll count it.", ["grep -c '<tool_call>' s.jsonl"]),
    ("live v1.10 reply", LIVE,
     "The user says session.jsonl contains a literal tag `<tool_call>` followed by `<arg_key>` markup. I'll run it.",
     "The file contains the literal markup `<tool_call>` instead of an actual tool call. I'll check the saved session "
     "file for that literal string.", ["grep -c '<tool_call>' /tmp/workspace/documents/session.jsonl"]),
    ("mention, close, no call", "The `<tool_call>` tag.</think>\n\nIt leaked as `<tool_call>` text.",
     "The `<tool_call>` tag.", "It leaked as `<tool_call>` text.", []),
    # what already worked and still does: plain calls, and calls inside the think block that end the reply (D5 rule)
    ("plain close, call", "Plan.</think>" + CALL, "Plan.", "", ["grep -c x s.jsonl"]),
    ("value holds the tag", "Plan.</think>" + GREP, "Plan.", "", ["grep -c '<tool_call>' s.jsonl"]),
    ("call ends the think block", "Plan: grep.\n" + CALL, "Plan: grep.", "", ["grep -c x s.jsonl"]),
    ("call holding the tag ends the think block", "Plan.\n" + GREP, "Plan.", "", ["grep -c '<tool_call>' s.jsonl"]),
    ("call, then close", "Plan: grep.\n" + CALL + "</think>", "Plan: grep.", "", ["grep -c x s.jsonl"]),
    ("two calls end the think block", "Plan.\n" + CALL + "\n" + GREP, "Plan.", "",
     ["grep -c x s.jsonl", "grep -c '<tool_call>' s.jsonl"]),
    ("call, then more reasoning", "Plan.\n" + CALL + "\nNo, wait.</think>Done.", "Plan.\n" + CALL + "\nNo, wait.",
     "Done.", []),
]


def run(mod, reply: str, hold: int | None, streamed: bool) -> tuple[str, str, list[str]]:
    """The CUDA server's path (``server.py``): split, hide calls from the streamed content, stream calls, end parse."""

    think = mod.ThinkSplit(hold)
    streamer = mod.GlmCallStreamer(BASH) if streamed else None
    shown_reasoning = shown_content = ""

    def visible(raw: str, finished: bool) -> tuple[str, str]:
        reasoning, answer = think(raw, finished)
        return reasoning, answer, mod.hide_tool_calls(answer, finished=finished)

    for n in range(1, len(reply)) if streamed else ():
        reasoning, _, content = visible(reply[:n], False)
        # what went out is never taken back
        assert reasoning.startswith(shown_reasoning), f"reasoning rewrote {shown_reasoning[-40:]!r}"
        assert content.startswith(shown_content), f"content rewrote {shown_content[-40:]!r} as {content[-40:]!r}"
        shown_reasoning, shown_content = reasoning, content
        if think.stream_from is not None:
            streamer.feed(reply[:n][think.stream_from:])
    reasoning, answer, _ = visible(reply, True)
    assert reasoning.startswith(shown_reasoning), f"final reasoning dropped {shown_reasoning[-40:]!r}"
    content, calls = mod.parse_tool_calls(answer, BASH)
    assert content.strip().startswith(shown_content.strip()), f"final content dropped {shown_content[-40:]!r}"
    if streamer is not None and think.stream_from is not None:
        streamer.feed(reply[think.stream_from:])
        streamed_calls = [*streamer.calls[:streamer.sent], *streamer.rest(BASH)]
        if streamed_calls:                       # the server sends the streamer's calls when it followed any
            assert [c["function"]["arguments"] for c in streamed_calls] == \
                   [c["function"]["arguments"] for c in calls or ()], "streamed calls differ from the end parser's"
    return reasoning, content, [json.loads(c["function"]["arguments"])["command"] for c in calls or ()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="patched TensorFold src/ (holds tensorfold/)")
    sys.path.insert(0, ap.parse_args().source_root)
    import tensorfold.cuda.reply_text as mod

    failed = 0
    for name, reply, want_reasoning, want_content, want_calls in CASES:
        for label, hold, streamed in (("whole", None, False), ("streamed", None, True),
                                      ("streamed+hold", mod.THINK_CALL_HOLD, True)):
            try:
                reasoning, content, calls = run(mod, reply, hold, streamed)
                problems = []
                if reasoning.strip() != want_reasoning.strip():
                    problems.append(f"reasoning ends {reasoning[-50:]!r}")
                if content.strip() != want_content.strip():
                    problems.append(f"content {content[:70]!r}")
                if calls != want_calls:
                    problems.append(f"calls {calls}")
            except AssertionError as exc:
                problems = [str(exc)]
            print(f"{'ok  ' if not problems else 'FAIL'} {name} [{label}]" + ("" if not problems else
                                                                            ": " + "; ".join(problems)))
            failed += bool(problems)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
