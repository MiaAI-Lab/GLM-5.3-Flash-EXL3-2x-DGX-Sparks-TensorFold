"""CPU-only regressions using the actual CUDA App.run and reply parser.

No weights, CUDA context, network, or executed tools. Every synthetic token is
a character, allowing every envelope/argument boundary to be fragmented.
"""
import json
import os
import time
from types import SimpleNamespace

os.environ['TF_GLM_STREAM_SMOOTH'] = '0'
os.environ['TF_GLM_LOOP_GUARD'] = '0'

from tensorfold.cuda.reply_text import hide_tool_calls, parse_tool_calls, ThinkSplit
from tensorfold.cuda.server import App, PreparedRequest

TOOLS = [{'type': 'function', 'function': {'name': 'write', 'parameters': {
    'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']}}}]


def call(value):
    return '<tool_call>write<arg_key>content</arg_key><arg_value>' + value + '</arg_value></tool_call>'


class Tok:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)

    def token_to_id(self, tag):
        return {'<tool_call>': 1000001, '<arg_key>': 1000002, '</arg_value>': 1000003,
                '</think>': 1000004}.get(tag)


class Engine:
    eos = (0,)
    concurrent = True
    tp = 4

    def __init__(self, raw, width, ended):
        self.raw, self.width, self.ended = raw, width, ended

    def generate(self, prompt, count, sampling, feed):
        ids = list(map(ord, self.raw)) + ([0] if self.ended else [])
        for i in range(0, len(ids), self.width):
            if feed(ids[i:i + self.width]):
                break
        return {'cached': 0, 'decode_s': .1, 'prefill_s': .01}


def run(raw, width=1, thinking=False, stream=True, single=False, ended=True):
    app = App.__new__(App)
    app.engine = Engine(raw, width, ended)
    app.tok = Tok()
    app.served = 'glm53'
    app.aliases = ()
    prepared = PreparedRequest([42], len(raw) + 2, TOOLS, thinking, None)
    events = []
    body = {'stream': stream, 'parallel_tool_calls': not single,
            'messages': [{'role': 'user', 'content': 'CPU synthetic fixture.'}]}
    result = app.run(body, True, lambda delta: events.append(delta) or True, prepared=prepared)
    if stream:
        events += [result['final']] + result['call_deltas']
        for i, c in enumerate(result['calls'] or []):
            if i >= result['calls_streamed']:
                events.append({'tool_calls': [{'index': i, **c}]})
    return result, events


checks = 0
for value in ('hello', 'before </tool_call> AFTER_IN_FILE',
              'a </tool_call> b </tool_call> c',
              '<tool_call>this is literal file content</tool_call>',
              'unicode λ 😀\nquotes " and backslash \\ </tool_call> tail'):
    raw = call(value)
    for width in (1, 2, 3, 7, 17, len(raw) + 1):
        for thinking in (False, True):
            for streaming in (False, True):
                for single in (False, True):
                    full = ('brief thought</think>' if thinking else '') + 'Before. ' + raw + ' After.'
                    result, events = run(full, width, thinking, streaming, single)
                    assert result['finish'] == 'tool_calls', result
                    assert len(result['calls']) == 1, result
                    assert json.loads(result['calls'][0]['function']['arguments']) == {'content': value}, result
                    assert result['content'] == 'Before.  After.', result
                    if streaming:
                        visible = ''.join(e.get('content', '') for e in events)
                        assert visible.strip() == 'Before.  After.', (width, value, visible)
                        assert len([c for e in events for c in e.get('tool_calls', [])]) == 1, events
                    checks += 1
    # Every cumulative character boundary must keep the payload hidden.
    for i in range(len(raw) + 1):
        assert hide_tool_calls(raw[:i], finished=False) == '', (i, value)
        checks += 1
    # ThinkSplit must hold/promote the whole call, not release the argument tail as reasoning.
    split = ThinkSplit(1024)
    for i in range(len(raw)):
        reason, answer = split('thought ' + raw[:i], False)
        assert not reason[len('thought '):] and not answer, (i, reason, answer)
        checks += 1
    reason, answer = split('thought ' + raw + '</think>', True)
    assert reason == 'thought ' and answer == raw
    checks += 1

# Two calls, final-parser fallback formats, normal prose, incomplete/truncated values.
for width in (1, 3, 17, 10000):
    raw = call('a </tool_call> b') + '\n' + call('second')
    result, events = run(raw, width)
    assert len(result['calls']) == 2 and result['content'] == '', result
    assert not ''.join(e.get('content', '') for e in events).strip(), events
    checks += 1
    for cut in (call('a </tool_call> b')[:-len('</arg_value></tool_call>')],
                '<tool_call>write<arg_key>content</arg_key><arg_value>cut'):
        result, events = run(cut, width, ended=False)
        assert not result['calls'] and result['finish'] == 'length', result
        assert not ''.join(e.get('content', '') for e in events), events
        checks += 1
for raw in ('<tool_call>{"name":"write","arguments":{"content":"JSON"}}</tool_call>',
            '<tool_call><function=write><parameter=content>Qwen</parameter></function></tool_call>'):
    content, calls = parse_tool_calls(raw, TOOLS)
    assert content == '' and len(calls) == 1, (content, calls)
    checks += 1

# Calls at the end of an unclosed think block, including the long-call commit path.
for value in ('before </tool_call> after', 'long payload ' * 200 + '</tool_call> end'):
    for width in (1, 7, 64, 10000):
        for single in (False, True):
            raw = 'thought ' + call(value)
            result, events = run(raw, width, thinking=True, single=single)
            assert len(result['calls']) == 1, result
            assert json.loads(result['calls'][0]['function']['arguments']) == {'content': value}, result
            assert not result['content'] and not ''.join(e.get('content', '') for e in events), result
            checks += 1

# Bound CPU work on a large cumulative answer (not an inference benchmark).
big = call(('source line </tool_call> tail\n' * 12000))
started = time.perf_counter()
assert hide_tool_calls(big, finished=True) == ''
_, parsed = parse_tool_calls(big, TOOLS)
assert len(parsed) == 1
elapsed = time.perf_counter() - started
assert elapsed < 3, elapsed
checks += 1
print(json.dumps({'passed': True, 'checks': checks, 'large_fixture_bytes': len(big.encode()),
                  'large_fixture_seconds': elapsed, 'scope': 'actual App.run with fake CPU token engine'}))
