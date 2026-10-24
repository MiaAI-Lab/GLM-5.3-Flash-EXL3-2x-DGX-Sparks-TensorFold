#!/usr/bin/env python3
"""Self-test of the port's client tools against tools/fake_server.py, on any machine (no GPU, no model, no Sparks):

    python3 -I tools/selftest.py [--only compile,creds,kinds,counters,exact,spill,resume,prefix] [--keep]

Each tool must PASS against a correct fake and FAIL against a fake with the fault it exists to catch (--mutate), so a
check that cannot fail is caught too. Exit 1 when any expectation is not met. Logs and receipts go to a temporary folder
(kept with --keep, or on a failure). From the DeepSeek recipe's port (wojo's fork, tools/selftest.py), for the tools
this recipe carries (no KV-shard, pair or runbook checks: those are DeepSeek engine's).
"""
import argparse
import os
import py_compile
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.dont_write_bytecode = True           # it imports checklib: no __pycache__ in tools/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable or "python3"
FAST = ["--prefill-us-per-token", "3", "--decode-ms-per-token", "1"]
results = []


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Fake:
    def __init__(self, work, name, *args):
        self.work, self.name, self.args = work, name, list(args)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.start()

    def start(self, *extra):
        self.log = open(os.path.join(self.work, f"fake-{self.name}.log"), "a")
        self.p = subprocess.Popen([PY, "-I", os.path.join(HERE, "fake_server.py"), "--port", str(self.port)]
                                  + self.args + list(extra), stdout=self.log, stderr=subprocess.STDOUT)
        end = time.monotonic() + 20
        while time.monotonic() < end:
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(self.url + "/health", timeout=2).read()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError(f"fake {self.name} did not start")

    def stop(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        self.log.close()

    def restart(self, *extra):
        self.stop()
        self.start(*extra)


def run(work, label, args, expect_ok=True, env=None):
    """Run a tool; it must exit 0 (expect_ok) or non-zero (a planted fault caught)."""
    log = os.path.join(work, label.replace(" ", "_").replace("/", "_") + ".log")
    t0 = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})),
                           timeout=900)
    ok = (p.returncode == 0) == expect_ok
    results.append((label, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}: exit {p.returncode} (expected {'0' if expect_ok else 'non-zero'}), "
          f"{time.monotonic() - t0:.1f}s, log {log}", flush=True)
    if not ok:
        with open(log) as f:
            print("      " + "      ".join(f.readlines()[-15:]), flush=True)
    return ok


def tool(name):
    return [PY, "-I", os.path.join(HERE, name)]


def t_exact(work):
    base = Fake(work, "exact-base", *FAST)
    cand = Fake(work, "exact-cand", *FAST)
    bad = Fake(work, "exact-conc-wrong", "--mutate", "conc-wrong", *FAST)
    weak = Fake(work, "exact-weak", "--mutate", "no-token-sha", *FAST)
    j = lambda n: os.path.join(work, n)  # noqa: E731
    try:
        run(work, "exact: record baseline (8 streams)", tool("exact.py") + ["record", "--url", base.url, "--out", j("x-base.json")])
        run(work, "exact: record candidate --like", tool("exact.py") + [
            "record", "--url", cand.url, "--out", j("x-cand.json"), "--like", j("x-base.json")])
        run(work, "exact: compare identical", tool("exact.py") + ["compare", j("x-base.json"), j("x-cand.json")])
        run(work, "exact: 4-stream record still works", tool("exact.py") + [
            "record", "--url", cand.url, "--out", j("x-cand4.json"), "--like", j("x-base.json"), "--streams", "4"])
        run(work, "exact: compare 8-stream baseline vs 4-stream candidate (extra cases are notes)", tool("exact.py") + [
            "compare", j("x-base.json"), j("x-cand4.json")])
        run(work, "exact: fault conc-wrong caught by record (solo != conc_5..8)", tool("exact.py") + [
            "record", "--url", bad.url, "--out", j("x-bad.json"), "--like", j("x-base.json")], expect_ok=False)
        run(work, "exact: fault conc-wrong caught by compare", tool("exact.py") + [
            "compare", j("x-base.json"), j("x-bad.json")], expect_ok=False)
        run(work, "exact: weak record (no token_sha) refused by compare", tool("exact.py") + [
            "record", "--url", weak.url, "--out", j("x-weak.json"), "--like", j("x-base.json")])
        run(work, "exact: compare refuses weak", tool("exact.py") + ["compare", j("x-base.json"), j("x-weak.json")],
            expect_ok=False)
        # a smaller window: the cases past it are left out, and compare notes them (not a mismatch)
        run(work, "exact: record at --context 65536 (len_64000, len_100000 left out)", tool("exact.py") + [
            "record", "--url", cand.url, "--out", j("x-cand64k.json"), "--like", j("x-base.json"), "--context", "65536"])
        expect("exact: the record lists what it left out",
               __import__("json").load(open(j("x-cand64k.json")))["meta"].get("over_window") == ["len_64000", "len_100000"],
               __import__("json").load(open(j("x-cand64k.json")))["meta"].get("over_window"))
        run(work, "exact: compare a full record with one at a smaller window (notes)", tool("exact.py") + [
            "compare", j("x-base.json"), j("x-cand64k.json")])
    finally:
        for f in (base, cand, bad, weak):
            f.stop()


def t_counters(work):
    """checklib reads the engine's shapes: /health prompt_cache.spill as an array of {rank, ...}, /metrics
    spill_<name>{rank="N"}; every TierCounts field per rank and over the ranks, from the fake and from literal text."""
    sys.path.insert(0, HERE)
    import checklib as C
    h = {"status": "ok", "prompt_cache": {"disk_lookups": 3, "spill": [
        {"rank": 0, "writes_committed": 2, "disk_free_bytes": 100, "no_space": 1},
        {"rank": 1, "writes_committed": 3, "disk_free_bytes": 50, "no_space": 0}]}}
    m = ('# HELP tensorfold:spill_writes_committed x\n# TYPE tensorfold:spill_writes_committed counter\n'
         'tensorfold:spill_writes_committed{rank="0"} 2\ntensorfold:spill_writes_committed{rank="1"} 3\n'
         'tensorfold:spill_disk_free_bytes{rank="0"} 100\ntensorfold:spill_disk_free_bytes{rank="1"} 50\n'
         'tensorfold:prompt_cache_disk_lookups 3\ntensorfold:kv_pool_usage_ratio{lane="0"} 0.5\n')
    want = {"spill.writes_committed": 5, "spill.writes_committed.rank0": 2, "spill.writes_committed.rank1": 3,
            "spill.disk_free_bytes": 50, "prompt_cache.disk_lookups": 3}
    for label, got in (("checklib: engine-shaped /health (spill array per rank)", C.health_counters(h)),
                       ("checklib: engine-shaped /metrics (spill_*{rank=N})", C.metrics_counters(C.parse_metrics(m)))):
        bad = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
        results.append((label, not bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {label}{': ' + str(bad) if bad else ''}", flush=True)
    # KV sharding's counters, read as they come (no field list: the GLM engine's are its shard port's)
    hk = {"status": "ok", "kv_shard": {"parts": 2, "carries": 3, "per_part": [{"part": 0, "free": 10},
                                                                             {"part": 1, "free": 20, "live": 1}]}}
    mk = ('# TYPE tensorfold:kv_shard_parts gauge\ntensorfold:kv_shard_parts 2\ntensorfold:kv_shard_carries 3\n'
          'tensorfold:kv_shard_part_free{part="0"} 10\ntensorfold:kv_shard_part_free{part="1"} 20\n'
          'tensorfold:kv_shard_part_live{part="1"} 1\n')
    wk = {"kv_shard.parts": 2, "kv_shard.carries": 3, "kv_shard.part0.free": 10, "kv_shard.part1.free": 20,
          "kv_shard.part1.live": 1}
    for label, got in (("checklib: /health kv_shard (per_part rows)", C.health_counters(hk)),
                       ("checklib: /metrics kv_shard_* and kv_shard_part_*{part=N}", C.metrics_counters(C.parse_metrics(mk)))):
        bad = {k: got.get(k) for k, v in wk.items() if got.get(k) != v}
        results.append((label, not bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {label}{': ' + str(bad) if bad else ''}", flush=True)
    # the window: prompts sized to CONTEXT (checklib.fit)
    ok = C.fit(131072) == 117964 and C.fit(131072, 48) == 117916 and C.capped(500000, 131072, 48) == 117916 \
        and C.capped(16000, 131072) == 16000 and C.over_window([100000, 120000], 131072) == [120000]
    expect("checklib: fit / capped / over_window size prompts to the window", ok,
           (C.fit(131072), C.fit(131072, 48), C.over_window([100000, 120000], 131072)))
    d0, d1 = os.path.join(work, "cnt-r0"), os.path.join(work, "cnt-r1")
    f = Fake(work, "counters", "--spill-dir", d0, "--spill-dir-rank1", d1, *FAST)
    try:
        srv = C.Server(f.url)
        keys = [f"spill.{n}{r}" for n in C.SPILL_FIELDS for r in ("", ".rank0", ".rank1")]
        for label, got in (("checklib: every spill.* counter from the fake's /health + /metrics", srv.counters()),
                           ("checklib: every spill.* counter from the fake's /metrics alone",
                            C.metrics_counters(C.parse_metrics(srv.get("/metrics", raw=True) or "")))):
            miss = [k for k in keys if k not in got]
            results.append((label, not miss))
            print(f"{'PASS' if not miss else 'FAIL'}  {label}{': missing ' + str(miss[:6]) if miss else ''}", flush=True)
    finally:
        f.stop()


def t_spill(work):
    j = lambda n: os.path.join(work, n)  # noqa: E731
    run(work, "spill: every item's hit resumes its seed's kept point (test_spill_items.py)", tool("test_spill_items.py"))
    d0, d1 = j("spill-r0"), j("spill-r1")
    base = Fake(work, "spill-base", *FAST)
    sp = ["--spill-dir", d0, "--spill-dir-rank1", d1, "--mem-points", "6", "--write-ms", "150"] + FAST
    cand = Fake(work, "spill-cand", *sp)
    rec, state = j("spill-base.json"), j("spill-state.json")
    dirs = ["--rank0-dir", d0, "--rank1-dir", d1, "--filler-tokens", "3000", "--commit-timeout", "30"]
    S = tool("spill_check.py")
    fakes = [base, cand]
    try:
        run(work, "spill: record baseline", S + ["record", "--url", base.url, "--out", rec, "--sizes", "16000,40000",
                                                 "--cancel-tokens", "60000", "--dual-tokens", "30000",
                                                 "--kill-tokens", "30000", "--floor-tokens", "20000"])
        run(work, "spill: main phase", S + ["run", "--url", cand.url, "--baseline", rec, "--phase", "main",
                                            "--receipt", j("r-spill-main.json")] + dirs)
        run(work, "spill: prerestart", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "prerestart", "--state", state] + dirs)
        cand.stop()
        run(work, "spill: drop rank 1's manifest while stopped", S + ["drop", "--state", state, "--rank1-dir", d1])
        cand.start()
        run(work, "spill: postrestart (missing-rank1 judged, --strict)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "postrestart", "--state", state, "--strict",
            "--receipt", j("r-spill-restart.json")] + dirs)
        kill = f"{PY} -c 'import os; os.kill({cand.p.pid}, 9)'"
        run(work, "spill: prekill (PIDs resolved first, kill -9 while K2's write is pending)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "prekill", "--state", state,
            "--resolve-cmd", "true", "--kill-cmd", kill] + dirs)
        cand.restart()
        run(work, "spill: postkill (the in-flight case exercised, --strict)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "postkill", "--state", state, "--strict",
            "--receipt", j("r-spill-kill.json")] + dirs)
        cand.stop()
        # the floor: the store's rule (floor + need + queued > free) on a simulated 1 GB disk, an empty store each time
        free = 10 ** 9
        for label, margin, extra, ok in (
                ("spill: free-disk floor just under free disk (refused on both ranks)", 10000,
                 ["--floor-margin-bytes", "10000"], True),
                ("spill: floor sizing guard (margin above F1's estimated write) caught", 10000,
                 ["--floor-margin-bytes", "200000"], False),
                ("spill: floor far under free disk (F1 admitted) caught", 400000, [], False)):
            f0, f1 = j(f"floor-{margin}-{len(extra)}-r0"), j(f"floor-{margin}-{len(extra)}-r1")
            ff = Fake(work, "spill-floor", "--spill-dir", f0, "--spill-dir-rank1", f1, "--disk-free-bytes", str(free),
                      "--spill-min-free-gib", repr((free - margin) / 2 ** 30), *FAST)
            fakes.append(ff)
            try:
                run(work, label, S + ["run", "--url", ff.url, "--baseline", rec, "--phase", "floor",
                                      "--rank0-dir", f0, "--rank1-dir", f1, "--commit-timeout", "10",
                                      "--bytes-per-position", "2", "--small-state-bytes", "1024",
                                      "--receipt", j(f"r-spill-floor-{margin}-{len(extra)}.json")] + extra, expect_ok=ok)
            finally:
                ff.stop()
        run(work, "spill: estimate prints F1's write", S + ["estimate", "--baseline", rec, "--bytes"])
        for mut, checks in (("restore-wrong", "evict"), ("no-disk", "evict"), ("ignore-crc", "corrupt-block,corrupt-small")):
            m0, m1 = j(f"spill-{mut}-r0"), j(f"spill-{mut}-r1")
            f = Fake(work, f"spill-{mut}", "--spill-dir", m0, "--spill-dir-rank1", m1, "--mem-points", "6",
                     "--mutate", mut, *FAST)
            try:
                run(work, f"spill: fault {mut} caught ({checks})", S + [
                    "run", "--url", f.url, "--baseline", rec, "--phase", "main", "--checks", checks,
                    "--rank0-dir", m0, "--rank1-dir", m1, "--filler-tokens", "3000", "--commit-timeout", "30"],
                    expect_ok=False)
            finally:
                f.stop()
        # GLM's tier: the refused MM is forgotten on every rank and written again by its own pass, a slow small state
        # (g3: its temporary still there when postrestart listed): "no temporaries" must wait for that write
        g0, g1, gst = j("spill-gf-r0"), j("spill-gf-r1"), j("spill-gf-state.json")
        f = Fake(work, "spill-presence-forgets", "--spill-dir", g0, "--spill-dir-rank1", g1, "--mem-points", "6",
                 "--presence-forgets", "--small-write-ms", "1500", *FAST)
        fakes.append(f)
        gdirs = ["--rank0-dir", g0, "--rank1-dir", g1, "--commit-timeout", "30", "--state", gst]
        run(work, "spill: presence-forgets prerestart", S + ["run", "--url", f.url, "--baseline", rec, "--phase",
                                                             "prerestart"] + gdirs)
        f.stop()
        run(work, "spill: presence-forgets drop", S + ["drop", "--state", gst, "--rank1-dir", g1])
        f.start()
        run(work, "spill: postrestart waits for MM's write again before judging temporaries (--strict)", S + [
            "run", "--url", f.url, "--baseline", rec, "--phase", "postrestart", "--strict",
            "--receipt", j("r-spill-gf.json")] + gdirs)
        f.stop()
        # a store that restores without rank 1's presence: missing-rank1 must catch it
        m0, m1, mst = j("spill-ip-r0"), j("spill-ip-r1"), j("spill-ip-state.json")
        f = Fake(work, "spill-ignore-presence", "--spill-dir", m0, "--spill-dir-rank1", m1, "--mem-points", "6",
                 "--mutate", "ignore-presence", *FAST)
        fakes.append(f)
        mdirs = ["--rank0-dir", m0, "--rank1-dir", m1, "--commit-timeout", "30", "--state", mst]
        run(work, "spill: ignore-presence prerestart", S + ["run", "--url", f.url, "--baseline", rec, "--phase",
                                                            "prerestart"] + mdirs)
        f.stop()
        run(work, "spill: ignore-presence drop", S + ["drop", "--state", mst, "--rank1-dir", m1])
        f.start()
        run(work, "spill: fault ignore-presence caught (missing-rank1)", S + [
            "run", "--url", f.url, "--baseline", rec, "--phase", "postrestart"] + mdirs, expect_ok=False)
    finally:
        for f in fakes:
            if f.p.poll() is None:
                f.stop()
    t_parked(work)


def expect(label, ok, detail):
    results.append((label, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}", flush=True)


def t_kinds(work):
    """checklib.kind_of: the store's files by kind, its temporaries as temps, and the recipe's marker at the spill root
    (start.sh's) as neither: spill_check's "no temporaries" checks must not count it."""
    sys.path.insert(0, HERE)
    import checklib as C
    cases = {"blocks/ab12.blk": "block", "points/k.st": "small", "points/k.man": "manifest", "points/k.ids": "ids",
             "points/k.man.tmp": "temp", "points/.k": "temp", ".lost": "temp", "LOCK": "other",
             C.SPILL_MARKER: "marker", "points/" + C.SPILL_MARKER: "temp"}
    got = {p: C.kind_of(p) for p in cases}
    bad = {p: (got[p], k) for p, k in cases.items() if got[p] != k}
    expect("checklib: kind_of names the store's files, its temporaries and only the exact root marker as the marker",
           not bad, bad or f"{len(cases)} names")
    listing = {C.SPILL_MARKER: [0, 0.0], "points/k.man": [10, 0.0], "points/k.man.tmp": [3, 0.0]}
    temps = [p for p in listing if C.kind_of(p) == "temp"]
    expect("checklib: a clean spill root with the marker has no temporaries but its real one", temps == ["points/k.man.tmp"], temps)


def t_parked(work):
    """The parked gauge (requests waiting on a restore: the one reading it and every coalesced one) is non-zero while a
    coalesced restore is in flight and back to 0 after, which spill_check's settled check waits for."""
    import json
    import threading
    j = lambda n: os.path.join(work, n)  # noqa: E731
    d0, d1 = j("parked-r0"), j("parked-r1")
    f = Fake(work, "spill-parked", "--spill-dir", d0, "--spill-dir-rank1", d1, "--write-ms", "10",
             "--restore-ms-per-block", "150", *FAST)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health():
        return json.loads(op.open(f.url + "/health", timeout=5).read()).get("prompt_cache") or {}

    def ask():
        body = json.dumps({"messages": [{"role": "user", "content": " ".join(f"w{i % 97}" for i in range(12000))}],
                          "max_tokens": 8}).encode()
        req = urllib.request.Request(f.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        return json.loads(op.open(req, timeout=120).read())

    try:
        ask()                                   # the seed: its point (10,240 tokens) goes to both ranks' disks
        end = time.monotonic() + 30
        while time.monotonic() < end:
            sp = health().get("spill") or []
            if sp and all(r["points"] >= 1 and not r["writes_pending"] for r in sp):
                break
            time.sleep(0.1)
        f.restart()                             # memory empty: the point is on disk only
        seen, out = [], []
        ts = [threading.Thread(target=lambda: out.append(ask())) for _ in range(2)]
        for t in ts:
            t.start()
        while any(t.is_alive() for t in ts):
            seen.append(health().get("parked") or 0)
            time.sleep(0.02)
        for t in ts:
            t.join()
        pc = health()
        shared, committed = pc.get("restores_shared") or 0, pc.get("disk_hits_committed") or 0
        expect("spill: a coalesced restore shows parked > 0 while in flight",
               len(out) == 2 and shared >= 1 and committed >= 1 and max(seen + [0]) >= 2,
               f"restores_shared={shared} disk_hits_committed={committed} parked max={max(seen + [0])} "
               f"over {len(seen)} readings, replies={len(out)}")
        expect("spill: parked back to 0 after (spill_check's settled reading)",
               pc.get("parked") == 0 and not pc.get("restores_active"),
               f"parked={pc.get('parked')} restores_active={pc.get('restores_active')}")
    finally:
        f.stop()


def t_resume(work):
    """resume_bench against a fake with the tier on: seed (cold, then from memory), a restart (memory empty), resume
    (from disk), report PASS; where a disk restore replies differently (restore-wrong) the report FAILs; sizes past the
    window are refused before any request."""
    j = lambda n: os.path.join(work, n)  # noqa: E731
    R = tool("resume_bench.py")
    for mut, ok in (("", True), ("restore-wrong", False)):
        tag = f" ({mut})" if mut else ""
        d0, d1, st = j(f"resume{mut}-r0"), j(f"resume{mut}-r1"), j(f"resume{mut}.json")
        f = Fake(work, f"resume{mut}", "--spill-dir", d0, "--spill-dir-rank1", d1, "--mem-points", "8",
                 *(["--mutate", mut] if mut else []), *FAST)
        common = ["--state", st, "--url", f.url, "--commit-timeout", "30", "--idle-wait", "5"]
        try:
            run(work, "resume: seed (cold, then from memory)" + tag, R + ["seed"] + common + [
                "--sizes", "14000,20000", "--reps", "1", "--warm-tokens", "12000"])
            f.restart()
            run(work, "resume: resume after a restart (from disk)" + tag, R + ["resume"] + common)
            run(work, "resume: report" + (f": fault {mut} caught" if mut else " PASS"), R + ["report"] + common + [
                "--md", j(f"resume{mut}.md")], expect_ok=ok)
        finally:
            f.stop()
    run(work, "resume: sizes past the window refused (no request sent)", R + [
        "seed", "--state", j("resume-over.json"), "--url", "http://127.0.0.1:9", "--sizes", "120000"], expect_ok=False)


def t_prefix(work):
    j = lambda n: os.path.join(work, n)  # noqa: E731
    P = tool("prefix_fanout.py")
    # --serial-prefill is the Zig engine's admission (one prompt pass at a time: a shared prefix is computed once);
    # without it the fake's prefills overlap and compute the prefix again, which --expect-one-fill must catch
    cases = [("prefix: record on a serial server, one fill at every level", ["--serial-prefill"],
              ["--out", j("fan-base.json"), "--system-tokens", "30000", "--expect-one-fill",
               "--receipt", j("r-fan-serial.json")], True),
             ("prefix: baseline replies again on a fresh serial server", ["--serial-prefill"],
              ["--baseline", j("fan-base.json"), "--expect-one-fill", "--strict-cold"], True),
             ("prefix: measure-only on a server whose prefills overlap", [],
              ["--baseline", j("fan-base.json"), "--receipt", j("r-fan-overlap.json")], True),
             ("prefix: duplicate prefill caught (--expect-one-fill, prefills overlap)", [],
              ["--baseline", j("fan-base.json"), "--expect-one-fill"], False),
             ("prefix: a warm server is not judged cold (--strict-cold)", ["--serial-prefill"], None, False)]
    for label, fake_args, args, ok in cases:
        f = Fake(work, "prefix", "--mem-points", "32", *fake_args, *FAST)
        grid = ["--grid", "2048"]      # the fake keeps on a 2048-token grid (GLM's is 64, the tool's default)
        try:
            if args is None:     # the same prompts twice on one server: the second pass is warm
                warm = ["--baseline", j("fan-base.json"), "--strict-cold"]
                run(work, label + " (first pass)", P + ["--url", f.url] + grid + warm)
                args = warm
            run(work, label, P + ["--url", f.url] + grid + args, expect_ok=ok)
        finally:
            f.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="compile,creds,kinds,counters,exact,spill,resume,prefix")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="tf-tools-selftest-")
    print(f"selftest: work folder {work}", flush=True)
    only = a.only.split(",")
    if "compile" in only:
        for f in sorted(os.listdir(HERE)):
            if f.endswith(".py"):
                try:
                    py_compile.compile(os.path.join(HERE, f), cfile=os.path.join(work, f + "c"), doraise=True)
                    results.append((f"py_compile {f}", True))
                except py_compile.PyCompileError as e:
                    results.append((f"py_compile {f}", False))
                    print(f"FAIL  py_compile {f}: {e}")
        print(f"PASS  py_compile of {sum(1 for n, _ in results if n.startswith('py_compile'))} tools"
              if all(ok for _, ok in results) else "FAIL  py_compile", flush=True)
    if "creds" in only:
        run(work, "checklib: API key rules (test_checklib_credentials.py)", tool("test_checklib_credentials.py"))
    for name, fn in (("kinds", t_kinds), ("counters", t_counters), ("exact", t_exact), ("spill", t_spill),
                     ("resume", t_resume), ("prefix", t_prefix)):
        if name in only:
            try:
                fn(work)
            except Exception as e:  # noqa: BLE001
                results.append((name, False))
                print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    bad = [n for n, ok in results if not ok]
    print(f"\nselftest: {'PASS' if not bad else 'FAIL'} ({len(results)} expectations, {len(bad)} unmet){': ' + ', '.join(bad) if bad else ''}")
    if not bad and not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"selftest: logs kept in {work}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
