#!/usr/bin/env python3
"""End-to-end checks of the spill tier (wojo's GLM 0088/0110/0111, ported to the GLM Zig engine) against a running
server: every resume that must come from disk is proven by the engine's counters, and every reply is compared by
strong token hash (tensorfold.token_sha) with a baseline record made on the unchanged image. From the DeepSeek recipe's
port (wojo's fork, tools/spill_check.py), without its picture checks (GLM's lane host keeps no picture prompts), with
GLM's block sizes and its prompts capped to the window (--context, CONTEXT, 131,072 by default). It needs the Sparks:
a pair serving with the spill tier on (SPILL_GIB, PROMPT_CACHE_GIB, SPILL_V2=1); tools/selftest.py runs it here against
tools/fake_server.py.

    # 1. on the BASELINE server (the unchanged image, spill off): the replies every check must reproduce
    python3 -I tools/spill_check.py record --out spill-base.json
    # 2. on the CANDIDATE server (spill on, a separate SPILL_DIR), phase by phase; the caller restarts between them
    python3 -I tools/spill_check.py run --baseline spill-base.json --phase main --receipt r-main.json \\
        --rank0-dir $SPILL_DIR --rank1-dir $SPILL_DIR --rank1-shell "ssh <worker>" --sudo
    python3 -I tools/spill_check.py run ... --phase prerestart --state S.json     # then: stop.sh, and while stopped:
    python3 -I tools/spill_check.py drop --state S.json --rank1-dir $SPILL_DIR --rank1-shell "ssh <worker>" --sudo
    python3 -I tools/spill_check.py run ... --phase postrestart --state S.json    # after start.sh
    python3 -I tools/spill_check.py run ... --phase prekill --state S.json --resolve-cmd "R" --kill-cmd "K"
    python3 -I tools/spill_check.py run ... --phase postkill --state S.json       # after the caller started it again
    python3 -I tools/spill_check.py estimate --baseline spill-base.json --item F1  # F1's write size, for the floor
    python3 -I tools/spill_check.py run ... --phase floor --floor-margin-bytes M   # floor = free disk - M, M < need

Phases and their checks (each PASS / FAIL / SKIP, SKIP = not exercised and why; exit 1 on any FAIL, or any SKIP with
--strict):
  main         seed every item (its reply == baseline; its point committed: spill.writes_committed), evict memory
               with fillers until every older kept state is gone (prompt_cache.evicted), then:
               evict           the hit restores from disk (disk_hits_committed +1, restored_tokens > 0), reply == base
               corrupt-block   one byte of a block flipped on one rank: no disk hit, restore refused/failed or a CRC
               corrupt-small   / size error counted, reply == baseline (recomputed). truncate-*: the file cut in half
               truncate-block  (default ranks: corrupt on rank 0, truncate on rank 1 when its directory is given)
               truncate-small
               remove-small    rank 1's small-state file removed while running (the store answers presence from
                               its in-memory index, so the read itself fails): no disk hit, read_failed /
                               restores_failed counted, reply == baseline
               cancel          a long restore's client leaves mid-restore: restores_cancelled +1, nothing left active,
                               the same request again == baseline
               dual-restore    two restores at once while a third stream decodes: both from disk, all three replies
                               == baseline; the decoder's worst inter-token gap is reported
               sizes           restore TTFT per size (--sizes): from disk, reply == baseline
               settled         no restore active, no parked request, no queued write left
  prerestart   seed R1 and MM and wait for both commits on every rank; their files go to --state
  drop         (server stopped) remove MM's manifest on rank 1, recorded in --state
  postrestart  every rank recovered its points; R1 restores from disk, reply == base; missing-rank1: MM, whose
               manifest rank 1 lost while stopped, is not restored (presence refused, or rank 1 holds fewer points),
               reply == baseline (recomputed); no temporaries once every rank's writes settle (MM's pass writes it again)
  prekill      seed K1 (committed); run --resolve-cmd (the PIDs, before K2); stream K2 and run --kill-cmd the moment
               its write is queued and pending (spill.writes_pending > 0), or print KILL NOW
  postkill     K1 restores from disk; K2 == baseline (from disk or recomputed; SKIP of the in-flight case when its
               write was no longer pending at the kill); no temporaries left on either rank
  floor        on an empty store with floor = free - --floor-margin-bytes, F1 (--floor-tokens, its estimated write
               larger than the margin) is refused on every rank (spill.no_space, nothing committed, no new files); its
               reply == baseline and memory still serves it

The rank directories are the HOST paths of the spill tier's folder on each Spark (SPILL_DIR, which start.sh mounts at
/spill for the engine's --snapshot-dir; the store makes native-spill-<compat>/rank<N>/ below it). --rank1-shell runs the file operations (list, corrupt, truncate, remove;
see checklib.FSOPS) on the worker; --sudo runs them under sudo -n (the containers write as root).
Operator-only options. --sudo, --rank0-shell / --rank1-shell, --resolve-cmd and --kill-cmd act on the Sparks
themselves and are for the operator of the pair under test, on a machine and account they control; never pass them
values from anyone else, and never wire them to a service. --sudo runs the file operations as root through `sudo -n`
(non-interactive: it fails rather than prompt; the tier's files are root's). --rankN-shell is a command prefix (e.g.
"ssh <worker>", split on spaces) under which the file operations run on that rank's host; the remote shell joins the
arguments into one command line, so they are quoted (shlex). --resolve-cmd and --kill-cmd are command lines run
through the local shell (subprocess with shell=True: pipes, $(...) and globs are the shell's); --kill-cmd is meant to
`kill -9` the serving ranks. Each is used only when given (off by default), and only by the phases that name it:
without them the checks that need them SKIP (or print KILL NOW for the operator to act on).
Counters: cache-tier's /health prompt_cache{..., spill[{rank, ...}, ...]} (or /metrics spill_<name>{rank="N"}), read
per rank (checklib: spill.<name>.rank<N>, spill.<name> over the ranks). Stdlib only.
"""
import argparse
import concurrent.futures
import json
import os
import secrets
import sys
import threading
import time
import subprocess

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
import checklib as C  # noqa: E402

Q_SEED = "Summarize the text above in one sentence."
Q_HIT = "List the three most frequent words in the text above."
ITEM_TOKENS = 16000          # past --spill-min-tokens (8192 by default): its kept point is written
PHASES = {
    "main": ["E1", "CB", "CS", "TB", "TS", "RS", "X1", "D1", "D2"],
    "prerestart": ["R1", "MM"], "postrestart": ["R1", "MM"],
    "prekill": ["K1", "K2"], "postkill": ["K1", "K2"],
    "floor": ["F1"],
}
CHECK_ITEM = {"evict": "E1", "corrupt-block": "CB", "corrupt-small": "CS", "truncate-block": "TB",
              "truncate-small": "TS", "remove-small": "RS", "cancel": "X1", "dual-restore": "D1"}
MAIN_CHECKS = ["evict", "corrupt-block", "corrupt-small", "truncate-block", "truncate-small", "remove-small",
               "cancel", "dual-restore", "sizes", "settled"]
# What one kept point writes on a rank (the GLM store, docs/glm-port/spill-store.md in the engine): 2048-token blocks of
# 12,705,792 B (6,204 B a position: the 11 DSA layers' latent rows, 528 B, and pooled index rows, 144 B a 4 positions),
# the last one a tail (GLM keeps on a 64-token grid), and the small state (the rings, recurrent and conv states, the
# drafter's): its size is round 2's, so SMALL_STATE_BYTES is a lower bound (low is the floor phase's safe side: F1 then
# needs at least the estimate); the store's need adds the ids (4 B a token) and the manifest.
BLOCK_TOKENS = 2048
BYTES_PER_POSITION = 6204
SMALL_STATE_BYTES = 1 << 20
GRID = 64                    # the GLM prompt cache's keep grid (cuda_native.zig CacheRules.grid)
KEEP_WINDOW = 128            # the point sits at a grid boundary at least this far before the prompt's end (an estimate)
DECODER = "DEC"
FAIL_KEYS = ["spill.crc_errors", "spill.size_errors", "prompt_cache.restores_failed", "spill.presence_refused",
             "spill.read_failed"]


def items(a, sizes):
    """{name: tokens}: every item any phase uses (the record holds them all), the sizes' Z<n> and the decoder."""
    it = {n: ITEM_TOKENS for ph in PHASES.values() for n in ph}
    it.update({"X1": a.cancel_tokens, "D1": a.dual_tokens, "D2": a.dual_tokens, "K2": a.kill_tokens,
               "F1": a.floor_tokens})
    for z in sizes:
        it[f"Z{z}"] = z
    it[DECODER] = 400
    return it


def doc(salt, name, tokens, seed, pad=0):
    """An item's text: `pad` (record's, in the baseline's meta.pads) adds that many words and renames its ref, so a
    padded item shares nothing with its unpadded try."""
    ref = f"{salt}-{name}" + (f"+{pad}" if pad else "")
    return f"[ref {ref}]\n" + C.words(C.words_for(tokens) + pad, seed + sum(map(ord, name)) * 7919)


def msgs(salt, name, tokens, seed, q, pad=0):
    if name == DECODER:
        return [{"role": "user", "content": f"[ref {salt}-dec] Write a long, detailed story about a lighthouse keeper "
                                            "who finds a map. Do not stop early."}]
    return [{"role": "user", "content": doc(salt, name, tokens, seed, pad) + "\n\n" + q}]


# An item's seed and hit diverge at their question, and the engine keeps the seed's state at its last keep-grid
# boundary (GLM: the last 64-token boundary before the prompt's end), so a seed ending a few tokens past a boundary keeps
# a point holding part of Q_SEED that its hit can never resume, from memory or disk (g2 and g3's D1: seed 37,963 kept
# at 37,952, its hit resumed 0 even in the cache-on record, so dual-restore read one point and prefilled D1 cold).
# record checks each item: the seed sent again resumes its kept point, and the hit must resume as far; when it does not,
# the item is padded by PAD_STEP words (about a third of the grid: past any question's tail) and recorded again.
PAD_STEP = 24
PAD_TRIES = 4


def estimate(tokens: int, bytes_per_position: int = BYTES_PER_POSITION, small: int = SMALL_STATE_BYTES) -> dict:
    """The write a prompt of about `tokens` makes on each rank (store.zig keep's need, from an empty store): its whole
    blocks and its tail, at bytes_per_position a position."""
    at = max(0, (tokens - KEEP_WINDOW) // GRID * GRID)
    blocks = -(-at // BLOCK_TOKENS)
    return {"tokens": tokens, "at": at, "blocks": blocks,
            "need_bytes": at * bytes_per_position + small + 4 * at + 256 + 80 * blocks}


def load_state(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_state(path, st):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st, f, indent=1, sort_keys=True)


def rank_dir(a, r):
    d = getattr(a, f"rank{r}_dir")
    return C.RankDir(d, getattr(a, f"rank{r}_shell"), a.sudo, r) if d else None


def drop(a) -> int:
    """With the server stopped: remove MM's manifest on rank 1 (the files prerestart recorded in --state)."""
    st = load_state(a.state)
    files = (st.get("files", {}).get(a.item) or {}).get("1") or {}
    man = sorted(p for p in files if C.kind_of(p) == "manifest")
    d = rank_dir(a, 1)
    if d is None or not man:
        C.log(f"drop: {'no --rank1-dir' if d is None else f'no manifest of {a.item} on rank 1 in {a.state}'}")
        st["dropped"] = {"item": a.item, "error": "nothing to drop"}
        save_state(a.state, st)
        return 1
    try:
        info = d.run("remove", path=man[-1])
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        C.log(f"drop: {e}")
        st["dropped"] = {"item": a.item, "error": str(e)}
        save_state(a.state, st)
        return 1
    st["dropped"] = dict(info, item=a.item, rank=1, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_state(a.state, st)
    C.log(f"drop: removed rank 1's {man[-1]} ({a.item})")
    return 0


def estimate_cmd(a) -> int:
    m = json.load(open(a.baseline))["meta"]
    e = estimate(int(m["items"][a.item]), a.bytes_per_position, a.small_state_bytes)
    print(e["need_bytes"] if a.bytes else json.dumps(e))
    return 0


# ---------------------------------------------------------------------- record
def record(a) -> int:
    srv = C.Server(a.url, a.model, a.timeout, a.seed)
    if not srv.alive():
        C.log(f"record: the server at {srv.url} is not healthy: {srv.health()}")
        return 2
    salt = a.salt or secrets.token_hex(6)
    sizes = [int(x) for x in a.sizes.split(",") if x]
    its = items(a, sizes)
    over = {n: t for n, t in its.items() if n != DECODER and t > C.fit(a.context, a.max_tokens)}
    if over:
        C.log(f"record: {over} do not fit the window ({a.context} tokens, CONTEXT or --context: at most "
              f"{C.fit(a.context, a.max_tokens)} a prompt)")
        return 2
    rows, pads, unverified, unresumable = {}, {}, [], []
    for name, tokens in its.items():
        if name == DECODER:
            rows[name] = {"seed": srv.chat(msgs(salt, name, tokens, a.seed, ""), a.decoder_tokens)}
        else:
            for t in range(PAD_TRIES):
                pad = t * PAD_STEP
                row = {"seed": srv.chat(msgs(salt, name, tokens, a.seed, Q_SEED, pad), a.max_tokens),
                       "hit": srv.chat(msgs(salt, name, tokens, a.seed, Q_HIT, pad), a.max_tokens)}
                again = srv.chat(msgs(salt, name, tokens, a.seed, Q_SEED, pad), a.max_tokens)
                if any("error" in x for x in (*row.values(), again)):
                    break
                kept, got = again.get("cached_tokens") or 0, row["hit"].get("cached_tokens") or 0
                if kept == 0:          # no cache on the record server (or nothing kept): it cannot be verified here
                    unverified.append(name)
                    break
                if got >= kept:
                    break
                C.log(f"  {name}: its hit resumed {got} of its seed's {kept} kept tokens (the kept point holds part "
                      f"of the question){'; padding it ' + str(PAD_STEP) + ' words' if t + 1 < PAD_TRIES else ''}")
            else:
                unresumable.append(name)
            rows[name] = row
            if pad:
                pads[name] = pad
        r = rows[name]["seed"]
        C.log(f"  {name}: " + (r.get("error") or f"prompt={r['prompt_tokens']} out={r['completion_tokens']} "
                                                  f"sha={r['token_sha'][:12]}({r['token_sha_source']}) {r['seconds']}s"
                                                  + (f" (padded {pads[name]} words)" if name in pads else "")))
    errors = [n for n, r in rows.items() if any("error" in x for x in r.values())] + unresumable
    weak = [n for n, r in rows.items() if any(x.get("token_sha_source") != "server" for x in r.values() if "error" not in x)]
    doc_ = {"meta": {"tool": "spill_check", "schema": 1, "salt": salt, "seed": a.seed, "model": a.model, "url": srv.url,
                     "context": a.context, "max_tokens": a.max_tokens, "decoder_tokens": a.decoder_tokens, "items": its, "sizes": sizes,
                     "pads": pads, "hits_unverified": unverified,
                     "label": a.label, "weak": weak, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "health": srv.health()}, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(doc_, f, indent=1, sort_keys=True)
    if weak:
        C.log(f"WARNING: no tensorfold.token_sha for {weak}: the record is WEAK (run needs --allow-weak)")
    if unverified:
        C.log(f"WARNING: {unverified}: a seed sent again resumed nothing (record on a server with the prompt cache on "
              "to check that each hit resumes its seed's kept point)")
    if unresumable:
        C.log(f"ERROR: {unresumable}: the hit never resumed its seed's kept point in {PAD_TRIES} paddings")
    C.log(f"wrote {a.out}: {len(rows)} items, errors {errors}")
    return 1 if errors else 0


# ---------------------------------------------------------------------- run
class Run:
    def __init__(self, a):
        self.a = a
        self.base = json.load(open(a.baseline))
        m = self.base["meta"]
        self.salt, self.seed, self.its = m["salt"], m["seed"], m["items"]
        self.pads = m.get("pads", {})
        self.srv = C.Server(a.url, m["model"] if not a.model_given else a.model, a.timeout, self.seed)
        self.mt, self.dt = m["max_tokens"], m["decoder_tokens"]
        self.rc = C.Receipt("spill_check", {"phase": a.phase, "url": self.srv.url, "baseline": a.baseline,
                                            "salt": self.salt, "label": a.label})
        self.ranks = {}
        if a.rank0_dir:
            self.ranks[0] = C.RankDir(a.rank0_dir, a.rank0_shell, a.sudo, 0)
        if a.rank1_dir:
            self.ranks[1] = C.RankDir(a.rank1_dir, a.rank1_shell, a.sudo, 1)
        self.files = {}

    def m(self, name, q):
        return msgs(self.salt, name, self.its[name], self.seed, q, self.pads.get(name, 0))

    def unresumable(self, name):
        """Why `name`'s hit cannot resume its seed's point at all (an old record's item whose kept point holds part
        of the question: its baseline hit, on a cache-on record, resumed nothing while others did), or None."""
        rows = self.base["rows"]
        if "pads" in self.base["meta"] or (rows[name]["hit"].get("cached_tokens") or 0) > 0:
            return None
        if not any((r.get("hit", {}).get("cached_tokens") or 0) > 0 for r in rows.values()):
            return None
        return (f"{name}: its hit resumed 0 in the baseline record (a cache-on record where other hits resumed): its "
                f"seed's kept point holds part of the question, so no tier can restore it; record again")

    def hit(self, name):
        return self.srv.chat(self.m(name, Q_HIT), self.mt)

    def same(self, name, kind, row):
        return C.same_reply(self.base["rows"][name][kind], row, self.a.allow_weak)

    def c(self):
        return self.srv.counters()

    def wait_commit(self, before, n=1):
        """Until every rank's writes_committed grew by n and no rank has a write pending (or the timeout); returns
        {rank: delta} (the aggregate as rank "all" when the server reports none per rank), {} when it reports none."""
        def per(c):
            d = C.rank_deltas(before, c, "writes_committed")
            if not d and "spill.writes_committed" in c:
                d = {"all": C.delta(before, c, "spill.writes_committed")}
            return d

        def ok():
            c = self.c()
            d = per(c)
            return c if d and min(d.values()) >= n and not c.get("spill.writes_pending") else None
        got = C.wait_until(ok, self.a.commit_timeout, 0.5)
        return per(got or self.c())

    def seed_item(self, name):
        """Send the item's seed request (== baseline), wait for its point, note its new files on every rank."""
        before = {r: d.list() for r, d in self.ranks.items()}
        c0 = self.c()
        row = self.srv.chat(self.m(name, Q_SEED), self.mt)
        ok, detail = self.same(name, "seed", row)
        committed = self.wait_commit(c0)
        self.files[name] = {r: d.new_files(before[r]) for r, d in self.ranks.items()}
        nf = {r: len(f) for r, f in self.files[name].items()}
        self.rc.check(f"seed {name} == baseline", ok, f"{detail}; committed per rank {committed or 'not reported'}; "
                      f"new files {nf}", row=row, committed=committed)
        return committed

    def committed_everywhere(self, committed):
        return bool(committed) and min(committed.values()) >= 1

    def evict(self, exclude_fill=0):
        """Fillers (own salts) until every kept state older than them is evicted (LRU), or --max-fillers."""
        c0 = self.c()
        entries = c0.get("prompt_cache.entries")
        target = int(entries) if entries is not None else None
        sent = 0
        while sent < self.a.max_fillers:
            name = f"fill-{secrets.token_hex(4)}"
            r = self.srv.chat([{"role": "user", "content": doc(self.salt + "-f", name, self.a.filler_tokens, self.seed)
                                + "\n\n" + Q_SEED}], 8)
            sent += 1
            if "error" in r:
                C.log(f"  filler {sent}: {r['error']}")
            d = C.delta(c0, self.c(), "prompt_cache.evicted")
            if target is None:
                if sent >= self.a.fillers:
                    break
            elif d is not None and d >= target:
                break
        d = C.delta(c0, self.c(), "prompt_cache.evicted")
        C.log(f"  eviction: {sent} fillers of {self.a.filler_tokens} tokens, evicted +{d} (entries before: {entries})")
        return {"fillers": sent, "evicted": d, "entries_before": entries,
                "complete": target is None or (d is not None and d >= target)}

    def files_of(self, name, rank, kind):
        f = self.files.get(name, {}).get(rank) or {}
        cands = sorted(p for p in f if C.kind_of(p) == kind)
        return cands[-1] if cands else None

    def disk_hit(self, name, check, extra=None):
        """Hit `name`: it must come from disk and equal the baseline."""
        c0 = self.c()
        row = self.hit(name)
        c1 = self.c()
        ok, detail = self.same(name, "hit", row)
        d = C.deltas(c0, c1, ["prompt_cache.disk_hits_committed", "prompt_cache.restored_tokens",
                              "prompt_cache.memory_hits", "prompt_cache.restore_wait_ns"] + FAIL_KEYS)
        dh = d["prompt_cache.disk_hits_committed"]
        if dh is None:
            return self.rc.add(check, "FAIL", "the server reports no prompt_cache.disk_hits_committed", row=row)
        from_disk = dh >= 1 and (d["prompt_cache.restored_tokens"] or 0) > 0
        why = [] if from_disk else [f"not from disk (disk_hits_committed +{dh}, memory_hits "
                                    f"+{d['prompt_cache.memory_hits']}: eviction incomplete? raise --max-fillers)"]
        if not ok:
            why.append(detail)
        u = self.unresumable(name)
        if u and not from_disk:
            why.append(u)
        self.rc.check(check, from_disk and ok, "; ".join(why) or f"from disk ({d['prompt_cache.restored_tokens']} "
                      f"tokens, cached {row.get('cached_tokens')}), {detail}", row=row, deltas=d, **(extra or {}))
        return from_disk and ok

    def fault(self, check, name, rank, kind, op):
        """Damage `name`'s file of `kind` on `rank` (op corrupt|truncate|remove), then hit: refused, recomputed."""
        d_ = self.ranks.get(rank)
        if d_ is None:
            return self.rc.add(check, "SKIP", f"no directory for rank {rank} (--rank{rank}-dir)")
        path = self.files_of(name, rank, kind)
        if path is None:
            return self.rc.add(check, "SKIP" if not self.files.get(name, {}).get(rank) else "FAIL",
                               f"no {kind} file of {name} found on rank {rank} (files: "
                               f"{sorted(self.files.get(name, {}).get(rank) or {})[:8]})")
        try:
            info = d_.run(op, path=path)
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            return self.rc.add(check, "FAIL", f"could not {op} {path} on rank {rank}: {e}")
        c0 = self.c()
        row = self.hit(name)
        c1 = self.c()
        ok, detail = self.same(name, "hit", row)
        d = C.deltas(c0, c1, ["prompt_cache.disk_hits_committed", "prompt_cache.restores_begun"] + FAIL_KEYS)
        refused = sum(v or 0 for k, v in d.items() if k in FAIL_KEYS)
        no_hit = (d["prompt_cache.disk_hits_committed"] or 0) == 0
        good = ok and no_hit and refused >= 1
        why = []
        if not no_hit:
            why.append(f"a damaged point was restored (disk_hits_committed +{d['prompt_cache.disk_hits_committed']})")
        if refused < 1:
            why.append("no refusal/CRC/size/presence/failed counter moved")
        if not ok:
            why.append(detail)
        self.rc.check(check, good, "; ".join(why) or f"{op} rank {rank} {path}: refused ({ {k: v for k, v in d.items() if v} }), "
                      f"{detail}", fault=dict(info, rank=rank, op=op), row=row, deltas=d)

    def cancel(self):
        name = "X1"
        c0 = self.c()
        ev = threading.Event()
        box = {}
        th = threading.Thread(target=lambda: box.update(row=self.srv.stream(self.m(name, Q_HIT), self.mt, cancel=ev)))
        th.start()

        def begun():      # leave at the first sign of the restore: begun, active, or blocks still to read
            c = self.c()
            return (C.delta(c0, c, "prompt_cache.restores_begun") or 0) >= 1 \
                or (c.get("prompt_cache.restores_active") or 0) >= 1 \
                or (c.get("prompt_cache.restore_blocks_done") or 0) < (c.get("prompt_cache.restore_blocks_total") or 0) \
                or not th.is_alive()
        seen = C.wait_until(begun, self.a.cancel_wait, 0.01)
        active = th.is_alive()
        if active and self.a.cancel_delay:
            time.sleep(self.a.cancel_delay)
        ev.set()
        th.join(60)
        settled = C.wait_until(lambda: (self.c().get("prompt_cache.restores_active", 0) == 0) or None, 120, 0.2)
        alive = self.srv.alive()
        c1 = self.c()
        d = C.deltas(c0, c1, ["prompt_cache.restores_cancelled", "prompt_cache.restores_begun",
                              "prompt_cache.disk_hits_committed", "spill.staging_bytes"])
        row = box.get("row") or {}
        again = self.hit(name)
        ok, detail = self.same(name, "hit", again)
        data = dict(deltas=d, cancelled_row=row, again=again, restore_seen=bool(seen) and active,
                    staging_bytes_after=c1.get("spill.staging_bytes"))
        if not alive or not settled:
            return self.rc.add("cancel", "FAIL", f"server alive={alive}, restores still active={not settled}", **data)
        if not ok:
            return self.rc.add("cancel", "FAIL", f"the request after the cancel: {detail}", **data)
        if (d["prompt_cache.restores_cancelled"] or 0) < 1:
            return self.rc.add("cancel", "SKIP", "not exercised: no restore was cancelled (it finished first, or none "
                               f"began: restores_begun +{d['prompt_cache.restores_begun']}); raise --cancel-tokens", **data)
        self.rc.add("cancel", "PASS", f"restore cancelled mid-flight (+{d['prompt_cache.restores_cancelled']}), nothing "
                    f"left active, the same request again: {detail}", **data)

    def dual(self):
        c0 = self.c()
        first = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            dec = ex.submit(self.srv.stream, self.m(DECODER, ""), self.dt, None, False, None, first)
            first.wait(self.a.timeout)
            t0 = time.monotonic()
            f1, f2 = ex.submit(self.hit, "D1"), ex.submit(self.hit, "D2")
            r1, r2 = f1.result(), f2.result()
            restores_s = time.monotonic() - t0
            rd = dec.result()
        c1 = self.c()
        d = C.deltas(c0, c1, ["prompt_cache.disk_hits_committed", "prompt_cache.restores_begun",
                              "prompt_cache.restore_wait_max_ns"])
        ok1, d1 = self.same("D1", "hit", r1)
        ok2, d2 = self.same("D2", "hit", r2)
        okd, dd = C.same_reply(self.base["rows"][DECODER]["seed"], rd, True)
        both = (d["prompt_cache.disk_hits_committed"] or 0) >= 2
        why = [] if both else [f"not both from disk (disk_hits_committed +{d['prompt_cache.disk_hits_committed']})"]
        why += [u for u in (self.unresumable("D1"), self.unresumable("D2")) if u]
        why += [f"D1: {d1}"] if not ok1 else []
        why += [f"D2: {d2}"] if not ok2 else []
        why += [f"decoder: {dd}"] if not okd else []
        gap = rd.get("max_gap")
        over = self.a.max_gap and gap is not None and gap > self.a.max_gap
        if over:
            why.append(f"decoder's worst inter-token gap {gap}s > --max-gap {self.a.max_gap}")
        self.rc.check("dual-restore", not why, "; ".join(why) or
                      f"both from disk in {restores_s:.2f}s while decoding; decoder == baseline ({dd}); its worst "
                      f"inter-token gap {gap}s", d1=r1, d2=r2, decoder=rd, deltas=d, restores_seconds=round(restores_s, 3))

    def sizes(self):
        names = [n for n in self.its if n.startswith("Z")]
        if not names:
            return self.rc.add("sizes", "SKIP", "the baseline has no --sizes items")
        out, bad = [], []
        for n in names:
            c0 = self.c()
            row = self.srv.stream(self.m(n, Q_HIT), self.mt)
            c1 = self.c()
            ok, detail = C.same_reply(self.base["rows"][n]["hit"], row, True)
            dh = C.delta(c0, c1, "prompt_cache.disk_hits_committed")
            cold = self.base["rows"][n]["seed"].get("seconds")
            out.append({"item": n, "tokens": self.its[n], "prompt_tokens": row.get("prompt_tokens"),
                        "restore_ttft": row.get("ttft"), "baseline_cold_seconds": cold, "from_disk": bool(dh),
                        "cached_tokens": row.get("cached_tokens"), "equal": ok,
                        "restore_wait_ns": C.delta(c0, c1, "prompt_cache.restore_wait_ns")})
            C.log(f"  {n}: ttft {row.get('ttft')}s from_disk={bool(dh)} (baseline cold {cold}s) {detail}")
            if not ok:
                bad.append(f"{n}: {detail}")
            elif not dh and self.its[n] >= self.a.spill_min_tokens + 2048 + 128:
                bad.append(f"{n}: not from disk")
        self.rc.check("sizes", not bad, "; ".join(bad) or f"{len(names)} sizes restored, replies equal", sizes=out)

    def settled(self):
        def ok():
            c = self.c()
            return c if not c.get("prompt_cache.restores_active") and not c.get("prompt_cache.parked") \
                and not c.get("spill.writes_pending") else None
        c = C.wait_until(ok, 120, 0.5)
        cc = c or self.c()
        self.rc.check("settled", bool(c) and self.srv.alive(),
                      f"restores_active={cc.get('prompt_cache.restores_active')} parked={cc.get('prompt_cache.parked')} "
                      f"writes_pending={cc.get('spill.writes_pending')} staging_bytes={cc.get('spill.staging_bytes')} "
                      f"staging_peak={cc.get('spill.staging_peak')}")

    def temporaries(self, check):
        """No temporary on any rank once this process's own writes have settled. A write in flight names its files
        `.<name>.<pid>.<seq>.tmp` until its rename (GLM: a refused point is forgotten on every rank and the prompt's pass
        writes it again, its ~100 MB small state for a while: g3 listed rank 0's mid-write), so the listing waits for
        every rank's writes_pending to reach 0 (--commit-timeout); a temporary left by an earlier process stays one."""
        settled = C.wait_until(lambda: not self.c().get("spill.writes_pending"), self.a.commit_timeout, 0.5)
        found = {}
        for r, d_ in self.ranks.items():
            try:
                found[r] = sorted(p for p in d_.list() if C.kind_of(p) == "temp")
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                return self.rc.add(check, "FAIL", f"could not list rank {r}: {e}")
        if not found:
            return self.rc.add(check, "SKIP", "no rank directory given")
        n = sum(len(v) for v in found.values())
        self.rc.check(check, n == 0, f"temporaries per rank: { {r: len(v) for r, v in found.items()} }"
                      + ("" if settled else f" (writes still pending after {self.a.commit_timeout}s)"), temporaries=found)

    # ------------------------------------------------------------------ phases
    def go(self) -> int:
        a = self.a
        if not self.srv.wait_healthy(a.health_timeout):
            self.rc.add("health", "FAIL", f"{self.srv.url} not healthy: {self.srv.health()}")
            return self.rc.finish(a.receipt)
        c0 = self.c()
        self.rc.doc["counters_before"] = c0
        if "prompt_cache.disk_lookups" not in c0 and "spill.points" not in c0:
            self.rc.add("tier on", "FAIL", "the server reports no disk tier (no prompt_cache.disk_lookups or spill.*): "
                        "started without --spill-dir/--spill-gib?")
            return self.rc.finish(a.receipt)
        getattr(self, "phase_" + a.phase)()
        self.rc.doc["counters_after"] = self.c()
        self.rc.doc["files"] = self.files
        return self.rc.finish(a.receipt, a.strict)

    def phase_main(self):
        a = self.a
        checks = a.checks.split(",") if a.checks else MAIN_CHECKS
        names = [CHECK_ITEM[c] for c in checks if c in CHECK_ITEM]
        if "dual-restore" in checks:
            names.append("D2")
        if "sizes" in checks:
            names += [n for n in self.its if n.startswith("Z")]
        for n in names:
            self.seed_item(n)
        ev = self.evict()
        self.rc.doc["eviction"] = ev
        if not ev["complete"]:
            C.log("WARNING: eviction incomplete: hits may come from memory (raise --max-fillers or --filler-tokens)")
        r_corrupt = a.fault_rank if a.fault_rank is not None else 0
        r_trunc = a.fault_rank if a.fault_rank is not None else (1 if 1 in self.ranks else 0)
        for chk in checks:
            try:
                if chk == "evict":
                    self.disk_hit("E1", "evict", {"eviction": ev})
                elif chk == "corrupt-block":
                    self.fault(chk, "CB", r_corrupt, "block", "corrupt")
                elif chk == "corrupt-small":
                    self.fault(chk, "CS", r_corrupt, "small", "corrupt")
                elif chk == "truncate-block":
                    self.fault(chk, "TB", r_trunc, "block", "truncate")
                elif chk == "truncate-small":
                    self.fault(chk, "TS", r_trunc, "small", "truncate")
                elif chk == "remove-small":
                    self.fault(chk, "RS", 1 if 1 in self.ranks else 0, "small", "remove")
                elif chk == "cancel":
                    self.cancel()
                elif chk == "dual-restore":
                    self.dual()
                elif chk == "sizes":
                    self.sizes()
                elif chk == "settled":
                    self.settled()
                else:
                    self.rc.add(chk, "FAIL", "unknown check")
            except Exception as e:  # noqa: BLE001 - one broken check must not hide the others
                self.rc.add(chk, "FAIL", f"{type(e).__name__}: {e}")

    def phase_prerestart(self):
        st = load_state(self.a.state)
        names = ("R1", "MM")
        for n in names:
            self.seed_item(n)
        st["files"] = {n: {str(r): f for r, f in self.files.get(n, {}).items()} for n in names}
        st.pop("dropped", None)
        save_state(self.a.state, st)
        C.log("prerestart: R1 and MM committed; now stop.sh, then `drop` (rank 1 loses MM's manifest), then start.sh, "
              "then --phase postrestart")

    def phase_postrestart(self):
        c = self.c()
        rp = C.rank_values(c, "recovered_points") or ({"all": c["spill.recovered_points"]}
                                                       if "spill.recovered_points" in c else {})
        self.rc.check("recovered points", bool(rp) and min(rp.values()) >= 1,
                      f"recovered_points per rank {rp or 'not reported'}")
        self.disk_hit("R1", "restart")
        self.missing_rank1()
        self.temporaries("no temporaries after a clean restart")

    def missing_rank1(self):
        """MM's manifest was removed on rank 1 while the server was stopped: rank 1 recovered without the point, so the
        all-rank presence check must refuse it and the reply be recomputed."""
        st = load_state(self.a.state)
        dr = st.get("dropped") or {}
        if dr.get("item") != "MM" or "error" in dr or "path" not in dr:
            return self.rc.add("missing-rank1", "SKIP", f"rank 1's manifest of MM was not removed while stopped "
                               f"(--state {self.a.state}: {dr or 'no drop recorded'}; run `spill_check.py drop`)")
        c0 = self.c()
        row = self.hit("MM")
        c1 = self.c()
        ok, detail = self.same("MM", "hit", row)
        d = C.deltas(c0, c1, ["prompt_cache.disk_hits_committed", "prompt_cache.restores_begun"] + FAIL_KEYS)
        pts = C.rank_values(c1, "points")
        fewer = len(pts) >= 2 and pts.get(1, 0) < pts.get(0, 0)
        refused = sum(v or 0 for k, v in d.items() if k in FAIL_KEYS)
        no_hit = (d["prompt_cache.disk_hits_committed"] or 0) == 0
        why = []
        if not no_hit:
            why.append(f"restored from disk though rank 1 lacks it (disk_hits_committed +{d['prompt_cache.disk_hits_committed']})")
        if refused < 1 and not fewer:
            why.append(f"no evidence rank 1 lacks it (no presence_refused/failed counter moved; points per rank {pts})")
        if not ok:
            why.append(detail)
        self.rc.check("missing-rank1", not why, "; ".join(why) or f"not restored (refused {refused}, points per rank "
                      f"{pts}), {detail}", dropped=dr, row=row, deltas=d)

    def phase_prekill(self):
        a = self.a
        st = load_state(a.state)
        self.seed_item("K1")
        if a.resolve_cmd:       # the PIDs, before K2: the kill must come while K2's write is still queued
            p = subprocess.run(a.resolve_cmd, shell=True, capture_output=True, text=True, timeout=120)
            if not self.rc.check("kill: PIDs resolved", p.returncode == 0,
                                 f"{a.resolve_cmd!r} exit {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}"):
                return
        c0 = self.c()
        box = {}
        th = threading.Thread(target=lambda: box.update(row=self.srv.stream(self.m("K2", Q_SEED), self.mt)))
        th.start()

        def queued():         # K2's write queued and not yet published on some rank
            c = self.c()
            q = C.delta(c0, c, "spill.writes_queued")
            return c if (q or 0) >= 1 and (c.get("spill.writes_pending") or 0) > 0 else None
        hit = C.wait_until(lambda: queued() or (None if th.is_alive() else "done"), a.commit_timeout, 0.01)
        at_kill = hit if isinstance(hit, dict) else self.c()
        pending = at_kill.get("spill.writes_pending")
        kinfo = {"writes_pending_at_kill": pending, "writes_pending_per_rank": C.rank_values(at_kill, "writes_pending"),
                 "committed_k2_before_kill": C.rank_deltas(c0, at_kill, "writes_committed"),
                 "reply_done_before_kill": not th.is_alive()}
        if a.kill_cmd:
            p = subprocess.run(a.kill_cmd, shell=True, capture_output=True, text=True, timeout=120)
            self.rc.check("kill", p.returncode == 0, f"{a.kill_cmd!r} exit {p.returncode} "
                          f"{(p.stdout + p.stderr).strip()[-200:]}; writes_pending at kill {pending}", **kinfo)
        else:
            print("KILL NOW: kill -9 both ranks, then restart and run --phase postkill", flush=True)
        th.join(a.timeout)
        st["prekill"] = kinfo
        save_state(a.state, st)
        self.rc.doc["prekill"] = kinfo
        if not pending:
            C.log("prekill: K2's write was no longer pending at the kill (postkill reports the in-flight case as SKIP)")

    def phase_postkill(self):
        st = load_state(self.a.state)
        c = self.c()
        self.rc.doc["recovery"] = {k: C.rank_values(c, k) or c.get("spill." + k)
                                   for k in ("recovered_points", "dropped_temporaries", "points")}
        self.disk_hit("K1", "kill: committed point restores")
        c0 = self.c()
        row = self.hit("K2")
        ok, detail = self.same("K2", "hit", row)
        dh = C.delta(c0, self.c(), "prompt_cache.disk_hits_committed")
        pk = st.get("prekill") or {}
        name = "kill: in-flight point hits or misses safely"
        how = f"{'from disk' if dh else 'recomputed'}; {detail}"
        if not ok:
            self.rc.add(name, "FAIL", how, row=row, prekill=pk)
        elif not pk.get("writes_pending_at_kill"):
            self.rc.add(name, "SKIP", f"reply equal ({how}), but K2's write was not pending at the kill "
                        f"({pk or 'no prekill state in --state'}): the queued-write case was not exercised",
                        row=row, prekill=pk)
        else:
            self.rc.add(name, "PASS", f"killed with {pk['writes_pending_at_kill']} write(s) pending; {how}",
                        row=row, prekill=pk)
        self.temporaries("no temporaries after recovery")

    def phase_floor(self):
        a = self.a
        est = estimate(int(self.its["F1"]), a.bytes_per_position, a.small_state_bytes)
        self.rc.doc["floor_estimate"] = dict(est, margin_bytes=a.floor_margin_bytes)
        if a.floor_margin_bytes and est["need_bytes"] <= a.floor_margin_bytes:
            self.rc.add("floor: sizing", "FAIL", f"F1's estimated write ({est['need_bytes']} B) is not above the margin "
                        f"left under free disk ({a.floor_margin_bytes} B): the floor would admit it; raise "
                        "--floor-tokens in the record or lower the margin", estimate=est)
        c0 = self.c()
        before = {r: d.list() for r, d in self.ranks.items()}
        row = self.srv.chat(self.m("F1", Q_SEED), self.mt)
        ok, detail = self.same("F1", "seed", row)
        C.wait_until(lambda: not self.c().get("spill.writes_pending"), self.a.commit_timeout, 0.5)
        time.sleep(1)
        c1 = self.c()
        d = C.deltas(c0, c1, ["spill.no_space", "spill.writes_committed", "spill.stored_bytes"])
        ns, wc = C.rank_deltas(c0, c1, "no_space"), C.rank_deltas(c0, c1, "writes_committed")
        new = {r: [p for p in d_.new_files(before[r]) if C.kind_of(p) in ("manifest", "block", "small")]
               for r, d_ in self.ranks.items()}
        if ns:
            refused = min(ns.values()) >= 1 and max(wc.values() or [0]) == 0
        else:
            refused = (d["spill.no_space"] or 0) >= 1 and (d["spill.writes_committed"] or 0) == 0
        nothing = not any(new.values())
        self.rc.check("floor", refused and nothing and ok,
                      f"no_space per rank {ns or d['spill.no_space']}, committed per rank {wc or d['spill.writes_committed']}, "
                      f"new files { {r: len(v) for r, v in new.items()} }, free per rank "
                      f"{C.rank_values(c1, 'disk_free_bytes') or c1.get('spill.disk_free_bytes')}, estimated need "
                      f"{est['need_bytes']} B, margin {a.floor_margin_bytes or '?'} B; {detail}",
                      row=row, deltas=d, no_space=ns, committed=wc, estimate=est)
        r2 = self.hit("F1")
        ok2, d2 = self.same("F1", "hit", r2)
        self.rc.check("floor: memory still serves", ok2, d2, row=r2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for p in (sub.add_parser("record"), sub.add_parser("run")):
        p.add_argument("--url", default=C.default_url())
        p.add_argument("--model", default=None)
        p.add_argument("--timeout", type=float, default=1800)
        p.add_argument("--label", default="")
    r = sub.choices["record"]
    r.add_argument("--out", required=True)
    r.add_argument("--salt", help="default: random (a cold record even on a warm server)")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--decoder-tokens", type=int, default=400, help="the decoding stream of dual-restore")
    r.add_argument("--context", type=int, default=C.default_context(),
                   help="the server's window (default CONTEXT, else 131072): every prompt must fit it, and the long "
                        "items below default to the most it allows")
    r.add_argument("--cancel-tokens", type=int, help="X1: long enough that its restore can be cut (default 500000, "
                                                     "capped to the window: ~0.73 GB to read at 131072, 6,204 B a token)")
    r.add_argument("--dual-tokens", type=int, default=40000)
    r.add_argument("--kill-tokens", type=int, help="K2: a write long enough to be killed in flight (default 200000, "
                                                   "capped to the window)")
    r.add_argument("--floor-tokens", type=int, help="F1: its write must exceed the floor suite's margin (default "
                                                    "200000, capped to the window; `estimate` prints its write)")
    r.add_argument("--sizes", default="16000,40000,100000", help="sizes check items (each must fit the window)")
    u = sub.choices["run"]
    u.add_argument("--baseline", required=True)
    u.add_argument("--phase", required=True, choices=sorted(PHASES))
    u.add_argument("--receipt", help="write the receipt JSON here")
    u.add_argument("--checks", help=f"main only, comma list (default all: {','.join(MAIN_CHECKS)})")
    u.add_argument("--rank0-dir")
    u.add_argument("--rank1-dir")
    u.add_argument("--rank0-shell", default="", help="operator-only: a command prefix the rank 0 file operations run under")
    u.add_argument("--rank1-shell", default="", help='operator-only: a command prefix the rank 1 file operations run under, '
                   'e.g. "ssh <worker>" (empty: rank 1 directory is local)')
    u.add_argument("--sudo", action="store_true", help="operator-only: the file operations as root under sudo -n")
    u.add_argument("--fault-rank", type=int, choices=(0, 1), help="corrupt and truncate on this rank (default: corrupt "
                                                                    "0, truncate 1)")
    u.add_argument("--max-fillers", type=int, default=120)
    u.add_argument("--fillers", type=int, default=24, help="when the server reports no prompt_cache.evicted")
    u.add_argument("--filler-tokens", type=int, default=32000)
    u.add_argument("--commit-timeout", type=float, default=300)
    u.add_argument("--health-timeout", type=float, default=900)
    u.add_argument("--cancel-wait", type=float, default=120, help="seconds to see the restore begin")
    u.add_argument("--cancel-delay", type=float, default=0, help="seconds after it began before leaving")
    u.add_argument("--max-gap", type=float, default=0, help="fail dual-restore above this decoder gap (0: report)")
    u.add_argument("--spill-min-tokens", type=int, default=8192, help="the server's, for the sizes check")
    u.add_argument("--kill-cmd", help="operator-only, run through the local shell: prekill's command that kills both ranks "
                   "(kill -9 by PID)")
    u.add_argument("--resolve-cmd", help="operator-only, run through the local shell: prekill's command before K2 (resolve "
                   "the PIDs, so the kill is one step)")
    u.add_argument("--state", help="prerestart/postrestart/prekill/postkill: the file the phases pass on")
    u.add_argument("--floor-margin-bytes", type=int, default=0, help="floor: free disk - floor (F1 must need more)")
    u.add_argument("--allow-weak", action="store_true")
    u.add_argument("--strict", action="store_true", help="a SKIP fails too")
    for p in (u, sub.add_parser("estimate", help="F1's (or --item's) estimated write per rank, from the record")):
        p.add_argument("--bytes-per-position", type=int, default=BYTES_PER_POSITION)
        p.add_argument("--small-state-bytes", type=int, default=SMALL_STATE_BYTES)
    e = sub.choices["estimate"]
    e.add_argument("--baseline", required=True)
    e.add_argument("--item", default="F1")
    e.add_argument("--bytes", action="store_true", help="print only the need in bytes")
    d = sub.add_parser("drop", help="(server stopped) remove MM's manifest on rank 1, recorded in --state")
    d.add_argument("--state", required=True)
    d.add_argument("--item", default="MM")
    d.add_argument("--rank1-dir", required=True)
    d.add_argument("--rank1-shell", default="", help="operator-only: a command prefix the removal runs under")
    d.add_argument("--sudo", action="store_true", help="operator-only: the removal as root under sudo -n")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    if a.mode == "estimate":
        return estimate_cmd(a)
    if a.mode == "drop":
        return drop(a)
    a.model_given = a.model is not None
    a.model = a.model or os.environ.get("MODEL", C.DEFAULT_MODEL)
    if a.mode == "record":
        for k, d in (("cancel_tokens", 500000), ("kill_tokens", 200000), ("floor_tokens", 200000)):
            if getattr(a, k) is None:
                setattr(a, k, C.capped(d, a.context, a.max_tokens, "--" + k.replace("_", "-")))
        return record(a)
    return Run(a).go()


if __name__ == "__main__":
    sys.exit(main())
