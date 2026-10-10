#!/usr/bin/env python3
"""Matched prose decode/TTFT screening; stdlib only, serial invocations on an idle server.

Poll /health for aggregate decode during full-concurrency windows, excluding
prefill and early/late tails. SSE timestamps measure client-observed TTFT.
Optional --prompts accepts a JSON string list for cold long-prompt tests;
use fresh prefixes and check cached_tokens, do not report cache hits as prefill.
No endpoint address, credentials, prompt text, or generated text is saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    p.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("tp4_prose_prompts.json"))
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), "Refusing to overwrite benchmark evidence"
    assert a.max_tokens > 0
    prompts = json.loads(a.prompts.read_text())
    assert len(prompts) >= a.concurrency and all(isinstance(x, str) and x for x in prompts)
    base = a.base_url.rstrip("/")
    def health():
        with urllib.request.urlopen(base+"/health", timeout=10) as r: return json.load(r)
    before = health()
    assert before.get("ok") and not before.get("busy"), "Server must be healthy and idle"
    barrier = threading.Barrier(a.concurrency)
    samples = []; stop = threading.Event()
    def monitor():
        while not stop.wait(1):
            try:
                h = health()
                samples.append(dict(t=time.monotonic(), decoding=h["streams"]["decoding"],
                                    tokens=h["completion_tokens_total"]))
            except (OSError, ValueError, KeyError) as exc:
                samples.append(dict(error=type(exc).__name__))
    def request(i):
        body = dict(model=a.model, messages=[dict(role="user", content=prompts[i])],
                    max_tokens=a.max_tokens, temperature=0, reasoning_effort="none", tf_policy="auto",
                    chat_template_kwargs=dict(enable_thinking=False), stream=True,
                    stream_options=dict(include_usage=True))
        req = urllib.request.Request(base+"/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        barrier.wait(timeout=20); start = time.monotonic()
        first = last = None; usage = metadata = finish = None; done = False
        digest = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "): continue
                data = line[6:].strip()
                if data == b"[DONE]": done = True; break
                event = json.loads(data)
                assert "error" not in event, "API stream error"
                usage = event.get("usage") or usage; metadata = event.get("tensorfold") or metadata
                for choice in event.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        last = time.monotonic()
                        if first is None: first = last
                        digest.update(text.encode())
        end = time.monotonic()
        assert done and first is not None and usage, "Incomplete SSE response"
        return dict(index=i, start=start, first=first, last=last, end=end, ttft_s=first-start,
                    usage=usage, finish_reason=finish, tensorfold=metadata,
                    prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(), output_sha256=digest.hexdigest())
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        with ThreadPoolExecutor(a.concurrency) as pool: results = list(pool.map(request, range(a.concurrency)))
    finally:
        stop.set(); thread.join(timeout=12)
    after = health(); tokens = sum(r["usage"]["completion_tokens"] for r in results)
    clean = (after["requests_total"]-before["requests_total"] == a.concurrency
             and after["completion_tokens_total"]-before["completion_tokens_total"] == tokens)
    steady = [s for s in samples if s.get("decoding") == a.concurrency
              and s["t"] > max(r["first"] for r in results)+3
              and s["t"] < min(r["last"] for r in results)-2]
    rate = ((steady[-1]["tokens"]-steady[0]["tokens"])/(steady[-1]["t"]-steady[0]["t"])
            if len(steady) >= 2 else None)
    report = dict(label=a.label, concurrency=a.concurrency, max_tokens=a.max_tokens,
        thinking=False, temperature=0, uncontaminated=clean, aggregate_steady_decode_tps=rate,
        mean_ttft_s=sum(r["ttft_s"] for r in results)/a.concurrency,
        output_tokens=tokens, requests=results, samples=samples,
        context_length=before.get("context_length"), parallel=before["streams"]["max"])
    with a.output.open("x") as file: json.dump(report, file, indent=2); file.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("requests", "samples")}), flush=True)
    assert clean, "Outside inference requests contaminated this run; do not compare it"


if __name__ == "__main__": main()
