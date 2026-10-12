"""Synthetic GLM reproducibility benchmark for the existing TensorFold API.
No restarts, config writes, private conversation reads, or model switching. Run only against an idle serving endpoint; arrange exclusivity outside this tool.
"""
import json
import time
import threading
import urllib.request
import statistics
import os
import argparse
MODEL=None

BASE = 'http://127.0.0.1:8888/v1'
HEALTH = 'http://127.0.0.1:8888/health'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def get_health():
    with opener.open(HEALTH, timeout=10) as r:
        return json.load(r)

def memory():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576

jobs = [
    ('prose-1', 'Write a detailed practical guide to maintaining a bicycle, including cleaning, inspecting brakes, adjusting gears, and fixing a flat. Use complete explanatory paragraphs.', False),
    ('code-1', 'Write a complete Python standard-library implementation of an LRU cache with TTL, thread safety, a fake clock, and unittest tests. Explain the invariants in comments.', False),
    ('code-repeat', 'Write a complete Python standard-library implementation of an LRU cache with TTL, thread safety, a fake clock, and unittest tests. Explain the invariants in comments.', False),
    ('prose-repeat', 'Write a detailed practical guide to maintaining a bicycle, including cleaning, inspecting brakes, adjusting gears, and fixing a flat. Use complete explanatory paragraphs.', False),
    ('synthetic-context', '\n'.join('Record %d: warehouse has %d blue bolts and %d green nuts.' % (i, i % 97, i % 83) for i in range(1800)) + '\nWrite a Python parser and tests for these records. Do not reproduce the records.', False),
    ('thinking-code', 'Design and implement a robust Python standard-library bounded work queue with cancellation, timeouts, and graceful shutdown. Analyze the race conditions before giving code.', True),
]

jobs = jobs + [(n+'-repeat2', p, t) for n,p,t in jobs[:2]]
def main(argv=None):
    global BASE, HEALTH, MODEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='the served model id, e.g. GLM-5.3-Flash-EXL3')
    parser.add_argument('--base-url', default=BASE)
    parser.add_argument('--health-url', default=HEALTH)
    parser.add_argument('--exclusive', action='store_true', help='Reject telemetry showing concurrent inference')
    args=parser.parse_args(argv)
    MODEL=args.model; BASE=args.base_url.rstrip('/'); HEALTH=args.health_url
    if args.exclusive: os.environ['BENCH_EXCLUSIVE']='1'
    results = []
    initial = get_health()
    assert initial['backend'] == 'tensorfold' and not initial['busy'], initial
    for name, prompt, thinking in jobs:
        before = get_health()
        if before['busy']:
            raise RuntimeError('Other work is active; refusing benchmark contention')
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0, 'max_tokens': 4096, 'stream': True,
                'stream_options': {'include_usage': True},
                'reasoning_effort': 'medium' if thinking else 'none',
                'chat_template_kwargs': {'enable_thinking': thinking}}
        samples = []
        stop = threading.Event()
        def monitor():
            while not stop.wait(0.5):
                try:
                    h = get_health()
                    samples.append({'t': time.monotonic(), 'tokens': h['completion_tokens_total'],
                                    'streams': h['streams'], 'requests': h['requests_running'],
                                    'mem_available_gib': memory()})
                except Exception as exc:
                    samples.append({'error': str(exc)})
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        start = time.monotonic()
        first = None
        chunks = 0
        visible = []
        reason_chars = 0
        usage = None
        models = set()
        done = False
        req = urllib.request.Request(BASE + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=600) as response:
                for raw in response:
                    if not raw.startswith(b'data: '):
                        continue
                    data = raw[6:].decode().strip()
                    if data == '[DONE]':
                        done = True
                        break
                    event = json.loads(data)
                    if event.get('model'):
                        models.add(event['model'])
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        content = delta.get('content') or ''
                        reasoning = delta.get('reasoning_content') or ''
                        if content or reasoning:
                            if first is None:
                                first = time.monotonic()
                            chunks += 1
                        visible.append(content)
                        reason_chars += len(reasoning)
        finally:
            end = time.monotonic()
            stop.set()
            thread.join(timeout=12)
        after = get_health()
        assert done and first and ''.join(visible) + ('x' if reason_chars else ''), name
        assert models == {MODEL}, models
        delta_keys = ('prompt_tokens_total', 'completion_tokens_total', 'prefill_seconds_total', 'decode_seconds_total', 'cached_tokens_total', 'drafted_total', 'accepted_total')
        deltas = {k: after[k] - before[k] for k in delta_keys}
        tokens = usage['completion_tokens'] if usage else deltas['completion_tokens_total']
        rates = [(b['tokens'] - a['tokens']) / (b['t'] - a['t']) for a, b in zip(samples, samples[1:]) if 't' in a and 't' in b]
        result = {'name': name, 'thinking': thinking, 'models': sorted(models), 'usage': usage,
                  'wall_seconds': end - start, 'first_generated_token_seconds': first - start,
                  'client_decode_tok_s': max(tokens - 1, 0) / (end - first),
                  'end_to_end_tok_s': tokens / (end - start), 'health_deltas': deltas,
                  'engine_active_decode_tok_s': deltas['completion_tokens_total'] / deltas['decode_seconds_total'] if deltas['decode_seconds_total'] else None,
                  'half_second_sample_peak_tok_s': max(rates) if rates else None,
                  'sse_content_events': chunks, 'reasoning_chars': reason_chars,
                  'visible_prefix': ''.join(visible)[:160], 'visible_output': ''.join(visible),
                  'min_available_gib': min([s['mem_available_gib'] for s in samples if 'mem_available_gib' in s], default=memory()),
                  'max_decoding_streams': max([s['streams']['decoding'] for s in samples if 'streams' in s], default=0),
                  'sampling_errors': [s for s in samples if 'error' in s]}
        results.append(result)
        if os.environ.get('BENCH_EXCLUSIVE') == '1':
            assert result['max_decoding_streams'] <= 1, 'Concurrent inference contaminated the sample'
            assert deltas['completion_tokens_total'] == tokens, 'Unowned tokens contaminated the sample'
        print(json.dumps({'run': result}), flush=True)
    print(json.dumps({'summary': {'model': MODEL, 'runs': len(results), 'median_client_decode_tok_s': statistics.median(r['client_decode_tok_s'] for r in results), 'max_client_decode_tok_s': max(r['client_decode_tok_s'] for r in results), 'initial_health': initial, 'final_health': get_health(), 'results': results}}), flush=True)

if __name__=='__main__': main()
