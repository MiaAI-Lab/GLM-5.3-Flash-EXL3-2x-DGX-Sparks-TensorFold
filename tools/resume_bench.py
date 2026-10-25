#!/usr/bin/env python3
"""Resume-from-spill bench at long prompts (up to the window: 32k-100k tokens by default at CONTEXT 131,072): the time
to first token of the same request computed cold, resumed from memory, and resumed from the spill tier on disk after a
restart, with every reply compared by strong token hash (tensorfold.token_sha) to the cold one. From the DeepSeek
recipe's port (wojo's fork, tools/resume_bench.py), sized for GLM-5.3-Flash on the Zig engine. It needs the Sparks: a
pair serving with the spill tier on (SPILL_GIB, PROMPT_CACHE_GIB); tools/selftest.py runs it here against
tools/fake_server.py.

    # 1. the server running with the spill tier on (memory may hold anything; every item gets its own salt)
    python3 -I tools/resume_bench.py seed --state S.json [--sizes 32000,64000,100000] [--reps 2]
    # 2. the caller restarts the server (memory empty; drop the page caches on every node while it is stopped)
    python3 -I tools/resume_bench.py resume --state S.json
    # 3. the table
    python3 -I tools/resume_bench.py report --state S.json [--md report.md]

seed    per rep, per size (ascending): `cold` streams the item's request (its own salt: nothing shared with any other
        prompt), waits until every rank committed its spill point (spill.writes_committed, nothing pending), then
        `memory` streams the identical request again (prompt_cache.memory_hits +1). A warm-up item W (--warm-tokens)
        is seeded first: the first restore after the restart pays the store's one-time costs on it, not on a bench row.
resume  a short unrelated request (the process's first-request costs), W from disk (not reported), then per rep, per
        size the identical request again: `disk` (prompt_cache.disk_hits_committed +1, restored_tokens > 0).
report  per size: prompt tokens, cold / memory / disk TTFT, cold / disk, restore wait, bytes read per rank, equality.

Every timed request is watched: /health live.connections is polled while it runs, and a row whose maximum exceeds 1
(another client in flight) is marked `overlap` and run again (--overlap-retries), since another client's work skews
the timing (it never changes the reply). Exit 1 when a reply differs from its cold one, a `memory` row did not come
from memory, a `disk` row did not come from disk, or a row stayed overlapped. Stdlib only; checklib for the client.
"""
import argparse
import json
import os
import secrets
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
import checklib as C  # noqa: E402

Q = "List the three most frequent words in the text above."
PC = "prompt_cache."
KEYS = [PC + k for k in ("memory_hits", "memory_misses", "disk_hits", "disk_hits_committed", "restored_tokens",
                         "restore_wait_ns", "restores_failed", "restores_refused", "restores_begun", "kept", "evicted",
                         "cached_tokens", "computed_tokens")]
RANK_KEYS = ("reads", "read_bytes", "read_failed", "crc_errors", "size_errors", "presence_refused", "writes_committed",
             "written_bytes", "writes_failed", "no_space")


def doc(salt, name, tokens, seed):
    return f"[ref {salt}-{name}]\n" + C.words(C.words_for(tokens), seed + sum(map(ord, name)) * 7919)


def msgs(salt, name, tokens, seed):
    return [{"role": "user", "content": doc(salt, name, tokens, seed) + "\n\n" + Q}]


class Watch:
    """Polls /health live.connections while a request runs; peak > 1 = another client was in flight."""

    def __init__(self, srv, every=0.5):
        self.srv, self.every, self.peak, self.waiting, self.stop = srv, every, 0, 0, threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.wait(self.every):
            h = self.srv.health()
            live = (h or {}).get("live") or {} if isinstance(h, dict) else {}
            self.peak = max(self.peak, int(live.get("connections") or 0))
            self.waiting = max(self.waiting, int(live.get("waiting") or 0))

    def __enter__(self):
        h = self.srv.health()
        live = (h or {}).get("live") or {} if isinstance(h, dict) else {}
        self.before = int(live.get("connections") or 0)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(5)


class Bench:
    def __init__(self, a):
        self.a = a
        self.st = json.load(open(a.state)) if os.path.exists(a.state) else {}

    def save(self):
        tmp = self.a.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, indent=1)
        os.replace(tmp, self.a.state)

    def server(self):
        m = self.st["meta"]
        return C.Server(self.a.url, m["model"], self.a.timeout, m["seed"])

    def m(self, it):
        meta = self.st["meta"]       # alt: a cold rerun's own prompt (attempt 0 kept its state: the same one would hit)
        return msgs(meta["salt"] + it.get("alt", ""), it["name"], it["tokens"], meta["seed"])

    def counters(self, srv):
        return srv.counters()

    def timed(self, srv, it, kind):
        """Stream the item's request, watched; rerun while overlapped. The row with counter deltas."""
        for attempt in range(self.a.overlap_retries + 1):
            if attempt:
                C.log(f"  {it['name']} {kind}: another client was in flight (peak {w.peak}); waiting, then again")
                if kind == "cold":
                    it["alt"] = f"~{attempt}"
                    self.save()
            if not C.wait_until(lambda: not self.busy(srv), self.a.idle_wait, 2):
                C.log(f"  {it['name']} {kind}: still busy after {self.a.idle_wait}s; running anyway")
            c0 = self.c0 = self.counters(srv)
            with Watch(srv) as w:
                row = srv.stream(self.m(it), self.st["meta"]["max_tokens"])
            time.sleep(1.3)          # rank 1's counters refresh about once a second
            c1 = self.counters(srv)
            d = C.deltas(c0, c1, KEYS)
            per = {k: C.rank_deltas(c0, c1, k) for k in RANK_KEYS}
            row.update({"kind": kind, "attempt": attempt, "deltas": d, "rank": per,
                        "peak_connections": w.peak, "peak_waiting": w.waiting, "connections_before": w.before,
                        "tail": (row.get("prompt_tokens") or 0) - (row.get("cached_tokens") or 0),
                        "overlap": w.before > 0 or w.peak > 1 or
                                   ((d.get(PC + "memory_hits") or 0) + (d.get(PC + "memory_misses") or 0)) > 1,
                        "at": time.strftime("%H:%M:%S")})
            if not row["overlap"] or kind == "warm":
                return row
            if kind == "disk" and (d.get(PC + "disk_hits_committed") or 0) >= 1:
                return row      # a second try would come from memory: keep it, marked
        return row

    def busy(self, srv):
        h = srv.health()
        live = (h or {}).get("live") or {} if isinstance(h, dict) else {}
        c = C.health_counters(h) if isinstance(h, dict) else {}
        return int(live.get("connections") or 0) > 0 or bool(c.get(PC + "restores_active")) or \
            bool(c.get("spill.writes_pending"))

    def wait_commit(self, srv, before):
        def ok():
            c = self.counters(srv)
            d = C.rank_deltas(before, c, "writes_committed")
            if not d and "spill.writes_committed" in c:
                d = {"all": C.delta(before, c, "spill.writes_committed")}
            return d if d and min(d.values()) >= 1 and not c.get("spill.writes_pending") else None
        t0 = time.monotonic()
        got = C.wait_until(ok, self.a.commit_timeout, 0.5)
        if got:                   # twice, 1.1 s apart: rank 1's pending count may be a second old
            time.sleep(1.1)
            got = ok() or got
        return got, round(time.monotonic() - t0, 2)

    def log_row(self, it, row):
        d = row["deltas"]
        C.log(f"  {it['name']:>10} {row['kind']:6s} ttft {row.get('ttft')}s prompt {row.get('prompt_tokens')} "
              f"cached {row.get('cached_tokens')} mem+{d.get(PC + 'memory_hits')} disk+{d.get(PC + 'disk_hits_committed')}"
              f" restored {d.get(PC + 'restored_tokens')} wait {round((d.get(PC + 'restore_wait_ns') or 0) / 1e9, 3)}s"
              f" read {row['rank'].get('read_bytes')} peak {row['peak_connections']}"
              + (f" ERROR {row['error']}" if "error" in row else ""))

    # ------------------------------------------------------------------ phases
    def seed(self):
        a = self.a
        sizes = [int(x) for x in a.sizes.split(",") if x]
        over = C.over_window(sizes + [a.warm_tokens], a.context, a.max_tokens)
        if over:
            C.log(f"--sizes / --warm-tokens {over} do not fit the window ({a.context} tokens, CONTEXT or --context: at "
                  f"most {C.fit(a.context, a.max_tokens)} a prompt)")
            return 2
        self.st = {"meta": {"salt": a.salt or secrets.token_hex(4), "seed": a.seed, "model": a.model,
                            "max_tokens": a.max_tokens, "context": a.context, "sizes": sizes, "reps": a.reps, "url": a.url,
                            "label": a.label, "seeded": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                   "items": [{"name": "W", "tokens": a.warm_tokens, "rep": -1, "warm": True}] +
                            [{"name": f"R{r}-{s}", "tokens": s, "rep": r} for r in range(a.reps) for s in sizes],
                   "rows": {}}
        srv = self.server()
        if not srv.wait_healthy(a.health_timeout):
            C.log("server not healthy")
            return 1
        h = srv.health()
        self.st["meta"]["health_before"] = {k: h.get(k) for k in ("kv_shard", "live") if isinstance(h, dict)}
        self.save()
        for it in self.st["items"]:
            n, rows = it["name"], {}
            rows["cold"] = self.timed(srv, it, "cold")
            self.log_row(it, rows["cold"])
            committed, waited = self.wait_commit(srv, self.c0)      # from the attempt that was kept
            rows["commit"] = {"per_rank": committed, "waited_s": waited}
            C.log(f"  {n:>10} commit {committed} after {waited}s")
            if "error" in rows["cold"] or not committed:
                self.st["rows"][n] = rows
                self.save()
                if it.get("warm") or not any(r.get("commit", {}).get("per_rank") for r in self.st["rows"].values()):
                    C.log(f"  {n}: cold failed or never committed ({rows['cold'].get('error')}): stopping, no restart")
                    return 1
                C.log(f"  {n}: cold failed or never committed: its disk row will fail")
            if not it.get("warm"):
                rows["memory"] = self.timed(srv, it, "memory")
                self.log_row(it, rows["memory"])
            self.st["rows"][n] = rows
            self.save()
        return 0

    def resume(self):
        a = self.a
        srv = self.server()
        if not srv.wait_healthy(a.health_timeout):
            C.log("server not healthy")
            return 1
        c = self.counters(srv)
        self.st["meta"]["after_restart"] = {"entries": c.get(PC + "entries"),
                                            "recovered_points": {r: c.get(f"spill.recovered_points.rank{r}")
                                                                 for r in (0, 1)},
                                            "resumed": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        first = {"name": f"first-{secrets.token_hex(3)}", "tokens": 600}
        r = srv.stream(msgs(self.st["meta"]["salt"] + "-x", first["name"], first["tokens"], self.st["meta"]["seed"]), 8)
        C.log(f"  first request after the restart: ttft {r.get('ttft')}s {r.get('error', '')}")
        for it in self.st["items"]:
            row = self.timed(srv, it, "warm" if it.get("warm") else "disk")
            self.log_row(it, row)
            self.st["rows"].setdefault(it["name"], {})["disk"] = row
            self.save()
        return 0

    def report(self):
        st, out, bad = self.st, [], []
        for it in st["items"]:
            if it.get("warm"):
                continue
            rows = st["rows"].get(it["name"], {})
            cold = rows.get("cold")
            for kind in ("cold", "memory", "disk"):
                if not rows.get(kind):
                    bad.append(f"{it['name']} {kind}: no row (phase not run or interrupted)")
            if cold and (cold.get("cached_tokens") or 0) > 0:
                bad.append(f"{it['name']} cold: not cold (cached {cold.get('cached_tokens')})")
            if cold and cold.get("overlap"):
                bad.append(f"{it['name']} cold: overlapped (peak {cold['peak_connections']})")
            for kind in ("memory", "disk"):
                row = rows.get(kind)
                if not cold or not row:
                    continue
                ok, detail = C.same_reply(cold, row)
                if not ok:
                    bad.append(f"{it['name']} {kind}: {detail}")
                d = row["deltas"]
                reused = (row.get("cached_tokens") or 0) > 0
                dh = d.get(PC + "disk_hits_committed") or 0
                src = reused and ((d.get(PC + "memory_hits") or 0) >= 1 and not dh if kind == "memory" else
                                  dh >= 1 and row.get("cached_tokens") == d.get(PC + "restored_tokens"))
                if not src:
                    bad.append(f"{it['name']} {kind}: not from {kind} ({detail})")
                if row.get("overlap"):
                    bad.append(f"{it['name']} {kind}: overlapped (peak {row['peak_connections']})")
        by = {}
        for it in st["items"]:
            if it.get("warm"):
                continue
            rows = st["rows"].get(it["name"], {})
            by.setdefault(it["tokens"], []).append(rows)
        lines = ["| prompt | cold TTFT | memory TTFT | disk TTFT | cold / disk | restore wait | read owner / other | "
                 "restored | tail | equal |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]

        def med(xs):
            xs = [x for x in xs if isinstance(x, (int, float))]
            return statistics.median(xs) if xs else None

        def f(x, nd=2, unit="s"):
            return "-" if x is None else f"{x:.{nd}f} {unit}".strip()

        def rng(xs, nd=2):
            xs = [x for x in xs if isinstance(x, (int, float))]
            if not xs:
                return "-"
            return f"{min(xs):.{nd}f}" if len(xs) == 1 or min(xs) == max(xs) else f"{min(xs):.{nd}f}-{max(xs):.{nd}f}"
        for tokens in sorted(by):
            rs = by[tokens]
            cold = [r.get("cold", {}).get("ttft") for r in rs]
            mem = [r.get("memory", {}).get("ttft") for r in rs]
            disk = [r.get("disk", {}).get("ttft") for r in rs]
            wait = [(r.get("disk", {}).get("deltas", {}).get(PC + "restore_wait_ns") or 0) / 1e9 for r in rs
                    if r.get("disk")]
            rb = [sorted((r.get("disk", {}).get("rank", {}).get("read_bytes") or {}).values(), reverse=True) for r in rs]
            r0 = med([x[0] for x in rb if x])                 # the owner reads the most (the rank differs by rep)
            r1 = med([x[1] for x in rb if len(x) > 1])
            tail = [r.get("disk", {}).get("tail") for r in rs]
            pt = med([r.get("cold", {}).get("prompt_tokens") for r in rs])
            restored = med([r.get("disk", {}).get("deltas", {}).get(PC + "restored_tokens") for r in rs])
            pairs = [(r.get("cold"), r.get(k)) for r in rs for k in ("memory", "disk")]
            eq = bool(pairs) and all(c and x and C.same_reply(c, x)[0] for c, x in pairs)
            mc, md_ = med(cold), med(disk)
            out.append({"tokens": tokens, "prompt_tokens": pt, "cold_ttft": cold, "memory_ttft": mem, "disk_ttft": disk,
                        "restore_wait_s": wait, "read_bytes_rank0": r0, "read_bytes_rank1": r1,
                        "restored_tokens": restored, "tail": tail, "speedup": (mc / md_) if mc and md_ else None, "equal": eq})
            lines.append(f"| {int(pt) if pt else tokens:,} | {rng(cold, 1)} s | {rng(mem)} s | {rng(disk)} s | "
                         f"{f'{mc / md_:.0f}x' if mc and md_ else '-'} | {rng(wait)} s | "
                         f"{(r0 or 0) / 1e6:.0f} / {(r1 or 0) / 1e6:.0f} MB | {int(restored or 0):,} | {rng(tail, 0)} | "
                         f"{'yes' if eq else 'NO'} |")
        md = "\n".join(lines)
        print(md)
        for b in bad:
            print("FAIL " + b)
        print(("PASS" if not bad else "FAIL") + f" resume_bench: {len(out)} sizes, {len(bad)} problems")
        self.st["report"] = {"sizes": out, "problems": bad, "table": md}
        self.save()
        if self.a.md:
            with open(self.a.md, "w") as fh:
                fh.write(md + "\n")
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("seed", "resume", "report"))
    ap.add_argument("--state", required=True, help="the JSON the phases pass on (and the results)")
    ap.add_argument("--url", default=C.default_url())
    ap.add_argument("--model", default=os.environ.get("MODEL", C.DEFAULT_MODEL))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--health-timeout", type=float, default=900)
    ap.add_argument("--commit-timeout", type=float, default=600)
    ap.add_argument("--sizes", default="32000,64000,100000", help="prompt sizes (tokens), each within the window")
    ap.add_argument("--context", type=int, default=C.default_context(),
                    help="the server's window (default CONTEXT, else 131072): seed refuses sizes that do not fit it")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warm-tokens", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--salt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label", default="")
    ap.add_argument("--overlap-retries", type=int, default=2)
    ap.add_argument("--idle-wait", type=float, default=300, help="seconds to wait for other clients before a rerun")
    ap.add_argument("--md", help="report: write the table here")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    b = Bench(a)
    if a.mode != "seed" and not b.st:
        C.log(f"no state in {a.state}: run seed first")
        return 2
    if a.mode != "seed":
        b.a.url = a.url
    return {"seed": b.seed, "resume": b.resume, "report": b.report}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
