#!/usr/bin/env python3
"""Common-system-prompt fan-out: N requests that share one long system prompt and differ only in their question, sent at
concurrency 1, 2, 4 and 8, each level on a fresh system prompt (its own salt) so every level starts cold. Per level it
counts what the engine did with the shared prefix:

  computed / cached tokens                  per request from usage.prompt_tokens_details.cached_tokens, and the
                                            prompt_cache.computed_tokens / cached_tokens counters when reported
  duplicate prefill tokens                  sum(computed) - ideal, where ideal = the first request's whole prompt plus,
                                            for each other request, its prompt less the shared cut (the system
                                            prompt's tokens rounded down to the keep grid, 64 tokens on GLM): the
                                            prefix computed more than once. Timing alone cannot prove one fill: this
                                            is the count.

    python3 -I tools/prefix_fanout.py --out fanout-base.json                     # baseline: record the replies
    python3 -I tools/prefix_fanout.py --baseline fanout-base.json --receipt r.json [--expect-one-fill]

PASS / FAIL per level: every request answered (and == baseline by strong token hash with --baseline); level 1
(sequential) must reuse the cut (duplicate <= --dup-tolerance per follower); with --expect-one-fill every level must.
Without it levels above 1 are measured and reported, not judged. The Zig engine admits one whole prompt pass at a time,
so a request sharing the prefix finds the one before it already kept: one fill at every level is expected, which is why
the Python fork's prefix wait (its 0110) is not ported (scripts/port.sh, PREFIX_WAIT). This tool is that claim's check:
--expect-one-fill on the Sparks, with the prompt cache on (PROMPT_CACHE_GIB). A level whose every request found the
prefix already cached was not cold (a rerun of the same --baseline prompts on a warm server): it is a SKIP, a failure
with --strict-cold. Run each pass on a freshly started server. From the DeepSeek recipe's port (wojo's fork,
tools/prefix_fanout.py). Stdlib only.
"""
import argparse
import concurrent.futures
import json
import os
import secrets
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checklib as C  # noqa: E402

QUESTIONS = ["Summarize the text above in one sentence.", "Which word comes last in the text above?",
             "Give a title for the text above.", "List three nouns from the text above.",
             "Name two words that repeat in the text above.", "Describe the text above in five words.",
             "Write one question a reader could ask about the text above.", "Pick one word from the text and define it.",
             "What is the first topic in the text above?", "Say one short sentence about the text above.",
             "Which word appears most often above?", "Reply with a number between one and nine."]
COUNTERS = ["prompt_cache.computed_tokens", "prompt_cache.cached_tokens", "prompt_cache.memory_hits",
            "chats_continued", "chats_continued_prefill_tokens"]


def level_messages(salt, level, n, sys_tokens, seed):
    system = f"[ref {salt}-c{level}] You are a careful assistant. Read the reference text, then answer briefly.\n" + \
        C.words(C.words_for(sys_tokens), seed + level * 7907)
    return [[{"role": "system", "content": system}, {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
            for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=C.default_url())
    ap.add_argument("--model", default=None)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--fanout", type=int, default=8, help="requests per level (all share the level's system prompt)")
    ap.add_argument("--system-tokens", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--salt")
    ap.add_argument("--grid", type=int, default=64, help="the prompt cache's keep grid (GLM: 64; the shared cut)")
    ap.add_argument("--context", type=int, default=C.default_context(), help="the server's window (default CONTEXT, "
                                                                             "else 131072)")
    ap.add_argument("--dup-tolerance", type=int, default=2048, help="duplicate tokens allowed per follower")
    ap.add_argument("--out", help="write a record (usable as --baseline)")
    ap.add_argument("--baseline", help="a record: reuse its salt and shapes, compare replies by token hash")
    ap.add_argument("--receipt")
    ap.add_argument("--expect-one-fill", action="store_true", help="every level must compute the prefix once")
    ap.add_argument("--allow-weak", action="store_true")
    ap.add_argument("--strict-cold", action="store_true", help="a level that was not cold (SKIP) fails the run")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--label", default="")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    base = json.load(open(a.baseline)) if a.baseline else None
    if base:
        bm = base["meta"]
        a.salt, a.seed, a.system_tokens, a.fanout, a.max_tokens = bm["salt"], bm["seed"], bm["system_tokens"], \
            bm["fanout"], bm["max_tokens"]
        a.model = a.model or bm["model"]
    a.model = a.model or os.environ.get("MODEL", C.DEFAULT_MODEL)
    salt = a.salt or secrets.token_hex(6)
    levels = [int(x) for x in a.levels.split(",") if x]
    if C.over_window([a.system_tokens + 64], a.context, a.max_tokens):
        C.log(f"--system-tokens {a.system_tokens} does not fit the window ({a.context} tokens, CONTEXT or --context)")
        return 2
    srv = C.Server(a.url, a.model, a.timeout, a.seed)
    rc = C.Receipt("prefix_fanout", {"url": srv.url, "salt": salt, "levels": levels, "fanout": a.fanout,
                                     "system_tokens": a.system_tokens, "baseline": a.baseline, "label": a.label})
    if not srv.wait_healthy(600):
        rc.add("health", "FAIL", f"{srv.url} not healthy: {srv.health()}")
        return rc.finish(a.receipt)
    rows_out, summary = {}, []
    for L in levels:
        msgs = level_messages(salt, L, a.fanout, a.system_tokens, a.seed)
        count, _, _ = srv.tokenize([msgs[0][0]])
        c0 = srv.counters()
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(L) as ex:
            rows = list(ex.map(lambda m: srv.chat(m, a.max_tokens), msgs))
        wall = time.monotonic() - t0
        c1 = srv.counters()
        d = C.deltas(c0, c1, COUNTERS)
        name = f"c{L}"
        rows_out[name] = rows
        errors = [r["error"] for r in rows if "error" in r]
        prompts = [r.get("prompt_tokens") or 0 for r in rows]
        cached = [r.get("cached_tokens") for r in rows]
        if count is None:             # no /tokenize: the shortest prompt less a question's worth
            count = min(prompts) - 48 if prompts else 0
        cut = max(0, (count - 16) // a.grid * a.grid)
        if any(c is None for c in cached) and d["prompt_cache.computed_tokens"] is not None:
            computed = d["prompt_cache.computed_tokens"]
        else:
            computed = sum(p - (c or 0) for p, c in zip(prompts, cached))
        ideal = sum(prompts) - (len(rows) - 1) * cut
        dup = computed - ideal
        info = {"level": L, "requests": len(rows), "wall_seconds": round(wall, 3), "system_tokens": count, "cut": cut,
                "computed_tokens": computed, "ideal_tokens": ideal, "duplicate_prefill_tokens": dup,
                "duplicate_per_follower": round(dup / max(1, len(rows) - 1), 1), "cached_tokens": cached,
                "seconds": [r.get("seconds") for r in rows], "deltas": d}
        summary.append(info)
        C.log(f"  c{L}: {len(rows)} requests in {wall:.2f}s, duplicate prefill {dup} tokens (cut {cut}), cached {cached}")
        why = []
        if errors:
            why.append(f"{len(errors)} errors: {errors[0]}")
        if base:
            brows = base["rows"].get(name) or []
            bad = [i for i, r in enumerate(rows)
                   if not C.same_reply(brows[i] if i < len(brows) else None, r, a.allow_weak)[0]]
            if bad:
                why.append(f"replies {bad} differ from the baseline: "
                           f"{C.same_reply(brows[bad[0]] if bad[0] < len(brows) else None, rows[bad[0]], a.allow_weak)[1]}")
        tol = a.dup_tolerance * max(1, len(rows) - 1)
        warm = bool(cached) and all(c for c in cached)   # even the first had the prefix: it was there before
        info["warm"] = warm
        judged = (L == 1 or a.expect_one_fill) and not warm
        if warm and not why:
            rc.add(f"fan-out c{L}", "SKIP", f"not cold: every request found the prefix cached ({cached[0]} tokens); run "
                   "on a freshly started server (a --baseline run reuses the record's prompts)", **info)
            continue
        if judged and dup > tol:
            why.append(f"the shared prefix was computed again: {dup} duplicate tokens > {tol}")
        rc.check(f"fan-out c{L}", not why, "; ".join(why) or
                 f"duplicate prefill {dup} tokens ({'judged' if judged else 'measured'}), {wall:.2f}s", **info)
    rc.doc["summary"] = summary
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"meta": {"tool": "prefix_fanout", "schema": 1, "salt": salt, "seed": a.seed, "model": a.model,
                                "system_tokens": a.system_tokens, "fanout": a.fanout, "max_tokens": a.max_tokens,
                                "levels": levels, "label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                       "rows": rows_out, "summary": summary}, f, indent=1, sort_keys=True)
        C.log(f"wrote {a.out}")
    return rc.finish(a.receipt, a.strict_cold)


if __name__ == "__main__":
    sys.exit(main())
