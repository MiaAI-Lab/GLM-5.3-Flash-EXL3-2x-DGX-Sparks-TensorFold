"""CPU-only regression for literal </think> inside GLM/Qwen/JSON tool-call values.

Blocked on upstream PR #134 and #158. Does not start a model or GPU.
"""
from __future__ import annotations

import json
import unittest

from tensorfold.cuda.reply_text import (
    GlmCallStreamer,
    ThinkSplit,
    _call_run,
    _think_close_outside_values,
    close_glm_call,
    hide_tool_calls,
    parse_tool_calls,
)

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def glm_call(command: str, tool_name: str = "exec_command") -> str:
    return (
        f"<tool_call>{tool_name}<arg_key>command</arg_key>"
        f"<arg_value>{command}</arg_value></tool_call>"
    )


class ToolValueThoughtBoundary(unittest.TestCase):
    def test_normal_glm_call_still_parses(self):
        text = glm_call("echo hello")
        self.assertEqual(_call_run(text, 0, True), (len(text), "calls"))
        content, calls = parse_tool_calls(text, TOOL)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["command"], "echo hello")

    def test_nested_think_close_in_glm_value_is_not_boundary(self):
        text = glm_call('echo "</think>"')
        self.assertEqual(_think_close_outside_values(text, 0, len(text)), -1)
        self.assertEqual(_call_run(text, 0, True), (len(text), "calls"))
        reasoning, answer = ThinkSplit()(text, True)
        self.assertEqual(reasoning, "")
        self.assertEqual(answer, text)

    def test_nested_tool_close_in_glm_value_is_not_envelope_end(self):
        text = glm_call('echo "</tool_call>" then done')
        content, calls = parse_tool_calls(text, TOOL)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["command"], 'echo "</tool_call>" then done')

    def test_both_literal_closers_in_glm_value_preserved(self):
        command = 'printf "%s" "</think> </tool_call>"'
        text = glm_call(command)
        self.assertEqual(_call_run(text, 0, True), (len(text), "calls"))
        self.assertEqual(hide_tool_calls(text, finished=True), "")
        content, calls = parse_tool_calls(text, TOOL)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["command"], command)

    def test_streaming_parser_keeps_nested_think_as_tool(self):
        text = glm_call('echo "</think>"')
        stream = ThinkSplit(hold=1024)
        stream(text[:29], False)
        stream(text[:58], False)
        reasoning, answer = stream(text, True)
        self.assertEqual((reasoning, answer), ("", text))
        parser = GlmCallStreamer(TOOL)
        deltas = parser.feed(answer)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["tool_calls"][0]["function"]["name"], "exec_command")

    def test_tool_tag_mentioned_in_prose_then_real_call(self):
        mention = "The literal <tool_call> tag is just documentation. "
        real = glm_call('echo "</think>"')
        content, calls = parse_tool_calls(mention + real, TOOL)
        self.assertEqual(content, mention.strip())
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["command"], 'echo "</think>"')
        self.assertEqual(hide_tool_calls(mention + real, finished=True), mention)

    def test_actual_reasoning_terminator_still_splits(self):
        tool = glm_call("echo done")
        reasoning = "Thinking through the request."
        text = reasoning + "</think>" + tool
        self.assertEqual(ThinkSplit()(text, True), (reasoning, tool))
        self.assertEqual(_think_close_outside_values(text, 0, len(text)), len(reasoning))

    def test_reasoning_tool_value_then_actual_terminator(self):
        tool = glm_call('echo "</think>"')
        text = "Reasoning text: " + tool + " and more reasoning</think>answer"
        self.assertEqual(_think_close_outside_values(text, 0, len(text)), text.rfind("</think>"))

    def test_qwen_parameter_value_literal_think_not_boundary(self):
        text = (
            '<tool_call><function=exec_command>'
            '<parameter=command>echo "</think>"</parameter>'
            '</function></tool_call>'
        )
        self.assertEqual(_think_close_outside_values(text, 0, len(text)), -1)
        self.assertEqual(_call_run(text, 0, True), (len(text), "calls"))

    def test_limit_truncated_arg_can_close_despite_literal_think(self):
        partial = '<tool_call>exec_command<arg_key>command</arg_key><arg_value>echo "</think>"'
        self.assertEqual(close_glm_call(partial, TOOL), '</arg_value></tool_call>')

    def test_limit_truncated_arg_can_close_despite_literal_tool_close(self):
        partial = '<tool_call>exec_command<arg_key>command</arg_key><arg_value>echo "</tool_call>"'
        self.assertEqual(close_glm_call(partial, TOOL), '</arg_value></tool_call>')

    def test_complete_call_not_closed_twice(self):
        self.assertEqual(close_glm_call(glm_call("echo done"), TOOL), "")

    def test_reasoning_prose_mentioning_arg_value_keeps_think_boundary(self):
        reasoning = 'The token <arg_value> is a markup example, not a tool argument.'
        text = reasoning + '</think>Answer is visible'
        self.assertEqual(ThinkSplit()(text, True), (reasoning, 'Answer is visible'))
        self.assertEqual(_think_close_outside_values(text, 0, len(text)), len(reasoning))

    def test_reasoning_prose_mentioning_qwen_parameter_keeps_think_boundary(self):
        reasoning = 'The tag <parameter=command> is only a documented example.'
        text = reasoning + '</think>Answer is visible'
        self.assertEqual(ThinkSplit()(text, True), (reasoning, 'Answer is visible'))

    def test_json_tool_argument_quoted_think_does_not_split(self):
        command = 'echo "</think>"'
        body = json.dumps({'name': 'exec_command', 'arguments': {'command': command}}, separators=(',', ':'))
        call = '<tool_call>' + body + '</tool_call>'
        self.assertEqual(_think_close_outside_values(call, 0, len(call)), -1)
        self.assertEqual(_call_run(call, 0, True), (len(call), 'calls'))
        self.assertEqual(ThinkSplit()(call, True), ('', call))
        content, calls = parse_tool_calls(call, TOOL)
        self.assertEqual(content, '')
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['command'], command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
