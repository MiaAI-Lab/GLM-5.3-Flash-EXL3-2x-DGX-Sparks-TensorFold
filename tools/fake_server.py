#!/usr/bin/env python3
"""A tiny stand-in for the GLM Zig server, so the port's checks can be self-tested on a laptop (tools/selftest.py).

    python3 -I tools/fake_server.py --port 18888 [--spill-dir D0 --spill-dir-rank1 D1] [--serial-prefill]
                                    [--context W] [--mutate NAME] ...

OpenAI-compatible /v1/chat/completions (non-streaming and SSE, greedy and deterministic: the reply depends only on the
messages and max_tokens), /v1/models (no max_model_len), /tokenize (count, max_model_len: the window), /health and
/metrics in the shapes the port's cache tier reports (the DeepSeek port's status_routes.zig putCache, metrics.zig
cacheFamilies, which the GLM cache core re-implements): prompt_cache{...} with `spill` an ARRAY of per-rank
{rank, <TierCounts>}, tensorfold:spill_<name>{rank="N"}, ...
From the DeepSeek recipe's port (wojo's fork, tools/fake_server.py), without its KV-shard parts and picture points.

What it simulates, with real files and real timing so the checks exercise their own logic:
- a prompt is its words (each a token) after a role tag; kept points sit at 2048-token boundaries with the rule
  `0 < at`, `at % grid == 0`, `at <= len - window`; a kept state is usable at any of its boundaries (a shared prefix);
- memory holds --mem-points kept states, least recently used evicted first (prompt_cache.evicted);
- the disk tier (with --spill-dir): written at keep by a writer thread under `<dir>/native-spill-fake/rank<N>/` in
  spill-store's layout (blocks/*.blk, points/*.{ids,st,man}, temporaries named `.<x>`, manifest last), one root per
  rank, each rank with its own counters and its own IN-MEMORY index (as store.zig: presence and the start of a read
  come from the index; a manifest file is read only at start, so removing one while running changes nothing until a
  restart). At start temporaries are dropped and each rank recovers its own manifests; a restore needs every rank's
  index to hold the point (else presence_refused), then reads both ranks' files block by block with
  --restore-ms-per-block (a file that cannot be opened: read_failed; wrong size: size_errors; CRC: crc_errors; each on
  the rank it happened, plus restores_failed), and is cancelled when its client leaves (restores_cancelled). The
  free-disk floor is the store's rule, per rank, at keep: refused (no_space) when floor + need + queued > free
  (--disk-free-bytes simulates a disk of that size, less what the rank stores);
- the Zig engine's admission (--serial-prefill): one whole prompt pass at a time, each looking the cache up again once
  the pass before it has kept its state, so requests sharing a prefix compute it once (without it, prefills overlap
  and a shared prefix is computed again beside the first: what prefix_fanout's --expect-one-fill must catch).
--mutate injects one fault so the checks can be shown to catch it: restore-wrong (a disk-restored reply differs),
no-disk (the tier never restores), ignore-crc (damaged blocks restore, wrongly), conc-wrong (replies differ while more
than four run at once), no-token-sha (no tensorfold.token_sha), ignore-presence (a restore reads a point some rank's
index lacks), admit-over-window (a prompt past the window is served).
Stdlib only. POST /fake/stop exits at once (os._exit, as a kill would) for restart tests.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import select
import socket
import threading
import time
import sys
import zlib
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
from checklib import SPILL_FIELDS  # noqa: E402  (the tier's counter names, one list)

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()
BLOCK_BYTES = 4096
SMALL_BYTES = 1024


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


class Gone(Exception):
    """The client closed its connection."""


class Fake:
    def __init__(self, a):
        self.a = a
        self.lock = threading.RLock()
        self.c = collections.Counter()           # prompt_cache counters
        self.t = [collections.Counter(), collections.Counter()]   # the tier's, per rank
        self.mem = collections.OrderedDict()     # id -> {"at", "hashes"}
        self.index = {}                          # boundary hash -> (id, at)
        self.next_id = 1
        self.disk = {}                           # point hash -> its manifest (rank 0's index: what lookups find)
        self.have = [set(), set()]               # each rank's in-memory index of published points
        self.stored = [0, 0]                     # bytes each rank stores
        self.queued = [0, 0]                     # bytes each rank has queued
        self.prefill_gate = threading.Lock()     # --serial-prefill: one prompt pass at a time
        self.active = 0
        self.slots = threading.Semaphore(a.parallel)
        self.writes = []                         # pending write jobs
        self.wcond = threading.Condition(self.lock)
        self.pending_restores = {}               # point hash -> event (coalesced restores)
        self.window = a.context
        if a.spill_dir:
            self.roots = [os.path.join(d, "native-spill-fake", f"rank{i}") for i, d in
                          enumerate([a.spill_dir, a.spill_dir_rank1 or a.spill_dir + "-rank1"])]
            self.recover()
            threading.Thread(target=self.writer, daemon=True).start()

    # ------------------------------------------------------------------ prompts
    def tokens(self, messages):
        out = []
        for m in messages:
            out.append(f"<{m.get('role')}>")
            c = m.get("content") or ""
            if isinstance(c, list):      # a picture is one token named by its payload's digest (the engine's spans)
                c = " ".join(p.get("text", "") if p.get("type", "text") == "text" else
                             "<picture:" + sha(json.dumps(p, sort_keys=True))[:16] + ">"
                             for p in c if isinstance(p, dict))
            out.extend(c.split())
        out.append("<assistant>")
        return out

    def boundaries(self, toks):
        """[(at, hash)] at every grid boundary the native rule allows for this prompt."""
        a, out, h = self.a, [], hashlib.sha256()
        limit = len(toks) - a.window
        for i, tk in enumerate(toks):
            h.update(tk.encode() + b"\0")
            at = i + 1
            if at % a.grid == 0 and at <= limit:
                out.append((at, h.copy().hexdigest()))
        return out

    def reply(self, toks, max_tokens, salt=""):
        ph = sha("\0".join(toks) + f"|{max_tokens}|{salt}")
        n = max(1, min(max_tokens, 12 + int(ph[:2], 16) % 20))
        return [WORDS[int(ph[(2 * i) % 60:(2 * i) % 60 + 2], 16) % len(WORDS)] for i in range(n)], ph

    # ------------------------------------------------------------------ memory
    def mem_best(self, bounds):
        best = 0
        for at, h in bounds:
            if h in self.index:
                best = at
        return best

    def mem_keep(self, bounds, at):
        hs = [h for b, h in bounds if b <= at]
        if not hs:
            return
        if hs[-1] in self.index:      # already kept: refresh
            i = self.index[hs[-1]][0]
            if i in self.mem and self.mem[i]["at"] >= at:
                self.mem.move_to_end(i)
                return
        i = self.next_id
        self.next_id += 1
        self.mem[i] = {"at": at, "hashes": hs}
        for b, h in bounds:
            if b <= at:
                self.index[h] = (i, b)
        self.c["kept"] += 1
        while len(self.mem) > self.a.mem_points:
            self.evict_one()

    def evict_one(self):
        if not self.mem:
            return False
        i, e = self.mem.popitem(last=False)
        for h in e["hashes"]:
            if self.index.get(h, (None,))[0] == i:
                del self.index[h]
        self.c["evicted"] += 1
        return True

    def mem_entry(self, bounds, at):
        for b, h in bounds:
            if b == at and h in self.index:
                return self.mem.get(self.index[h][0])
        return None

    # ------------------------------------------------------------------ disk tier
    def point_dir(self, r):
        return os.path.join(self.roots[r], "points")

    def recover(self):
        """Each rank on its own, as store.zig's open: temporaries deleted, its manifests parsed into its index."""
        for r, root in enumerate(self.roots):
            for sub in ("blocks", "points"):
                os.makedirs(os.path.join(root, sub), exist_ok=True)
                for f in os.listdir(os.path.join(root, sub)):
                    path = os.path.join(root, sub, f)
                    if f.startswith("."):
                        os.remove(path)
                        self.t[r]["dropped_temporaries"] += 1
                    else:
                        self.stored[r] += os.path.getsize(path)
            for f in os.listdir(self.point_dir(r)):
                if f.endswith(".man"):
                    try:
                        man = json.load(open(os.path.join(self.point_dir(r), f)))
                    except (OSError, ValueError):
                        continue
                    self.have[r].add(man["hash"])
                    self.t[r]["recovered_points"] += 1
                    if r == 0:
                        self.disk[man["hash"]] = man
                    else:
                        self.disk.setdefault(man["hash"], man)

    def free_bytes(self, r=0):
        if self.a.disk_free_bytes:
            return max(0, self.a.disk_free_bytes - self.stored[r])
        s = os.statvfs(self.roots[r])
        return s.f_bavail * s.f_frsize

    def need_bytes(self, r, hashes):
        """What a point would add on rank r (store.zig keep: new blocks, small state, ids, header, 80 B a block)."""
        new = sum(BLOCK_BYTES for bh in hashes
                  if not os.path.exists(os.path.join(self.roots[r], "blocks", bh[:32] + ".blk")))
        return new + SMALL_BYTES + 65 * len(hashes) + 64 + 80 * len(hashes)

    def enqueue_write(self, bounds, at):
        if not self.a.spill_dir or at < self.a.spill_min_tokens:
            return
        h = dict((b, x) for b, x in bounds)[at]
        if h in self.disk or any(w["hash"] == h for w in self.writes):
            return
        hashes = [x for b, x in bounds if b <= at]
        floor = self.a.spill_min_free_gib * 2 ** 30
        ranks, needs = [], {}
        for r in range(2):
            need = self.need_bytes(r, hashes)
            if floor + need + self.queued[r] > self.free_bytes(r):     # store.zig floorShort, on an empty store
                self.t[r]["no_space"] += 1
                continue
            ranks.append(r)
            needs[r] = need
            self.queued[r] += need
            self.t[r]["writes_queued"] += 1
        if not ranks:
            return
        self.writes.append({"hash": h, "at": at, "blocks": hashes, "ranks": ranks, "needs": needs})
        self.c["writes_enqueued"] += 1
        self.wcond.notify()

    def write_file(self, r, path, data):
        d, n = os.path.split(path)
        tmp = os.path.join(d, "." + n + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
        with self.lock:
            self.stored[r] += len(data)
            self.t[r]["written_bytes"] += len(data)

    def writer(self):
        while True:
            with self.lock:
                while not self.writes:
                    self.wcond.wait()
                job = self.writes[0]
            time.sleep(self.a.write_ms / 1000)
            man = {"hash": job["hash"], "at": job["at"], "blocks": [], "small": None}
            for bh in job["blocks"]:
                data = (bh * (BLOCK_BYTES // 64 + 1)).encode()[:BLOCK_BYTES]
                man["blocks"].append([bh[:32], len(data), zlib.crc32(data)])
            small = (job["hash"] * (SMALL_BYTES // 64)).encode()
            man["small"] = [len(small), zlib.crc32(small)]
            for r in job["ranks"]:
                for bh in job["blocks"]:
                    p = os.path.join(self.roots[r], "blocks", bh[:32] + ".blk")
                    if not os.path.exists(p):
                        self.write_file(r, p, (bh * (BLOCK_BYTES // 64 + 1)).encode()[:BLOCK_BYTES])
                    else:
                        with self.lock:
                            self.t[r]["dedup_blocks"] += 1
                self.write_file(r, os.path.join(self.point_dir(r), job["hash"][:32] + ".ids"), " ".join(job["blocks"]).encode())
                self.write_file(r, os.path.join(self.point_dir(r), job["hash"][:32] + ".st"), small)
                time.sleep(self.a.write_ms / 4000)
                self.write_file(r, os.path.join(self.point_dir(r), job["hash"][:32] + ".man"), json.dumps(man).encode())
                with self.lock:
                    self.have[r].add(job["hash"])
                    self.queued[r] -= job["needs"][r]
                    self.t[r]["writes_committed"] += 1
            with self.lock:
                self.writes.pop(0)
                self.disk[job["hash"]] = man

    def pending(self, r):
        return sum(1 for w in self.writes if r in w["ranks"] and w["hash"] not in self.have[r])

    def disk_best(self, bounds, beyond):
        """Rank 0's lookup: its index, in memory (no file is opened)."""
        best = None
        for at, h in bounds:
            if at > beyond and h in self.disk and h in self.have[0]:
                best = (at, h)
        return best

    def check_point(self, man, check_gone, flags):
        """The all-rank presence check (each rank's index), then both ranks' files block by block: returns
        (counter, rank) of the failure or None."""
        a = self.a
        for r in range(2):
            if man["hash"] not in self.have[r] and a.mutate != "ignore-presence":
                return "presence_refused", 0
        for r in range(2):
            p = os.path.join(self.point_dir(r), man["hash"][:32] + ".st")
            try:
                small = open(p, "rb").read()
            except OSError:
                return "read_failed", r
            if len(small) != man["small"][0]:
                return "size_errors", r
            if zlib.crc32(small) != man["small"][1]:
                if a.mutate != "ignore-crc":
                    return "crc_errors", r
                flags["corrupt"] = True
        for i, (name, size, crc) in enumerate(man["blocks"]):
            check_gone()
            time.sleep(a.restore_ms_per_block / 1000)
            with self.lock:
                self.c["restore_blocks_done"] += 1
            for r in range(2):
                try:
                    data = open(os.path.join(self.roots[r], "blocks", name + ".blk"), "rb").read()
                except OSError:
                    return "read_failed", r
                if len(data) != size:
                    return "size_errors", r
                if zlib.crc32(data) != crc:
                    if a.mutate != "ignore-crc":
                        return "crc_errors", r
                    flags["corrupt"] = True
        with self.lock:
            for r in range(2):
                self.t[r]["reads"] += 1
        return None

    def forget(self, h):
        """A damaged point is dropped on every rank (its files and its index entries)."""
        man = self.disk.pop(h, None)
        for r in range(2):
            self.have[r].discard(h)
        if man:
            for r in range(2):
                for ext in (".man", ".st", ".ids"):
                    try:
                        os.remove(os.path.join(self.point_dir(r), h[:32] + ext))
                    except OSError:
                        pass

    # ------------------------------------------------------------------ one request
    def run(self, messages, max_tokens, check_gone, emit=None):
        """Admit, restore, prefill and decode one request; returns the reply dict (emit gets each token)."""
        a = self.a
        toks = self.tokens(messages)
        if (len(toks) + 1 > self.window or len(toks) + min(max_tokens, 32) > self.window + 32) \
                and a.mutate != "admit-over-window":
            raise ValueError(f"This server's maximum context length is {self.window} tokens, but the rendered prompt "
                             f"has {len(toks)} tokens and leaves no room for a reply, which exceeds the context window.")
        bounds = self.boundaries(toks)
        restored, flags = False, {}
        with self.lock:
            best = self.mem_best(bounds)
            if best:
                self.c["memory_hits"] += 1
                e = self.mem_entry(bounds, best)
                if e:
                    self.mem.move_to_end(next(i for i, x in self.mem.items() if x is e))
            else:
                self.c["memory_misses"] += 1
            disk = None
            if a.spill_dir and a.mutate != "no-disk":
                self.c["disk_lookups"] += 1
                disk = self.disk_best(bounds, best)
                if disk is None:
                    self.t[0]["compatible_misses"] += 1
        if disk:
            at, h = disk
            with self.lock:
                self.c["disk_hits"] += 1
                waiter = self.pending_restores.get(h)
                if waiter is None:
                    ev = threading.Event()
                    self.pending_restores[h] = ev
                    self.c["restores_begun"] += 1
                    self.c["restores_active"] += 1
                    self.c["restore_blocks_total"] += len(self.disk[h]["blocks"])
                    man = self.disk[h]
                else:
                    self.c["restores_shared"] += 1
                # parked: requests waiting on a restore, the one that reads it and every coalesced one (the engine's
                # loading + deferred); each leaves it on every way out (done, failed, cancelled)
                self.c["parked"] += 1
            if waiter is not None:
                try:
                    waiter.wait(600)
                finally:
                    with self.lock:
                        self.c["parked"] -= 1
                with self.lock:
                    best = self.mem_best(bounds)
            else:
                t0 = time.monotonic()
                why = None
                try:
                    why = self.check_point(man, check_gone, flags)
                except Gone:
                    with self.lock:
                        self.c["restores_cancelled"] += 1
                        self.c["restores_active"] -= 1
                        self.c["restore_blocks_total"] -= len(man["blocks"])
                        self.c["restore_blocks_done"] = 0
                        self.c["parked"] -= 1
                        self.pending_restores.pop(h).set()
                    raise
                except BaseException:
                    with self.lock:
                        self.c["parked"] -= 1
                    raise
                with self.lock:
                    self.c["parked"] -= 1
                    self.c["restores_active"] -= 1
                    self.c["restore_blocks_total"] -= len(man["blocks"])
                    self.c["restore_blocks_done"] = 0
                    wait_ns = int((time.monotonic() - t0) * 1e9)
                    self.c["restore_waits"] += 1
                    self.c["restore_wait_ns"] += wait_ns
                    self.c["restore_wait_max_ns"] = max(self.c["restore_wait_max_ns"], wait_ns)
                    if why:
                        self.t[why[1]][why[0]] += 1
                        self.c["restores_failed"] += 1
                        if why[0] != "presence_refused":     # a rank that lacks it: the others keep theirs
                            self.forget(h)
                    else:
                        self.c["disk_hits_committed"] += 1
                        self.c["restored_tokens"] += at
                        self.mem_keep(bounds, at)
                        best = at
                        restored = True
                    self.pending_restores.pop(h).set()
        # --serial-prefill: the Zig engine's admission, one prompt pass at a time; the cache is looked up again once
        # the pass before this one has kept its state (released after the keep, or when the request ends)
        gated = a.serial_prefill and self.prefill_gate.acquire()
        try:
            if gated:
                with self.lock:
                    best = max(best, self.mem_best(bounds))
            # prefill what is not cached
            with self.lock:
                self.c["cached_tokens"] += best
                self.c["computed_tokens"] += len(toks) - best
            time.sleep((len(toks) - best) * a.prefill_us_per_token / 1e6)
            check_gone()
            with self.lock:
                keep_at = bounds[-1][0] if bounds else 0
                if keep_at:
                    self.mem_keep(bounds, keep_at)
                    self.enqueue_write(bounds, keep_at)
            if gated:            # the next prompt pass may start: decode runs beside it
                gated = False
                self.prefill_gate.release()
            # decode
            salt = ""
            if (restored and a.mutate == "restore-wrong") or flags.get("corrupt"):
                salt = "restored-differently"
            with self.lock:
                self.active += 1
                if self.active > 4 and a.mutate == "conc-wrong":
                    salt = "crowded"
            try:
                out, ph = self.reply(toks, max_tokens, salt)
                found = re.findall(r"SPECIAL ENTRY: (\w+) secret value is (\d+)", " ".join(toks))
                if found:       # a needle prompt: answer it, as a model that read the whole prompt would
                    out = json.dumps({k: v for k, v in found}).split(" ")
                with self.slots:
                    for tk in out:
                        check_gone()
                        time.sleep(a.decode_ms_per_token / 1000)
                        if emit:
                            emit(tk + " ")
            finally:
                with self.lock:
                    self.active -= 1
        finally:
            if gated:
                self.prefill_gate.release()
        text = " ".join(out) + " "
        tf = {"cached_tokens": best, "prefill_seconds": round((len(toks) - best) * a.prefill_us_per_token / 1e6, 4)}
        if a.mutate != "no-token-sha":
            tf["token_sha"] = sha(text)
        return {"text": text, "prompt_tokens": len(toks), "completion_tokens": len(out), "cached": best,
                "finish": "length" if len(out) >= max_tokens else "stop", "tensorfold": tf}

    # ------------------------------------------------------------------ status
    def cache_stats(self):
        on = bool(self.a.spill_dir or self.a.serial_prefill)
        if not on:
            return None
        pc = {k: self.c[k] for k in ("memory_hits", "memory_misses", "kept", "evicted", "refused", "failed",
                                      "disk_lookups", "disk_hits", "disk_hits_committed", "restores_begun",
                                      "restores_shared", "restores_failed", "restores_cancelled", "restores_dropped",
                                      "restores_active", "restore_blocks_done", "restore_blocks_total",
                                      "restored_tokens", "restore_waits", "restore_wait_ns", "restore_wait_max_ns",
                                      "writes_enqueued", "parked",
                                      "cached_tokens", "computed_tokens")}
        pc["entries"] = len(self.mem)
        if self.a.spill_dir:      # the engine's shape: one object per rank, rank first
            ranks = []
            for r in range(2):
                sp = {"rank": r}
                sp.update({k: self.t[r][k] for k in SPILL_FIELDS})
                sp.update(points=len(self.have[r]), stored_bytes=self.stored[r], cap_bytes=200 * 2 ** 30,
                          disk_free_bytes=self.free_bytes(r), writes_pending=self.pending(r),
                          blocks=len(os.listdir(os.path.join(self.roots[r], "blocks"))))
                ranks.append(sp)
            pc["spill"] = ranks
        return pc

    def health(self):
        h = {"status": "ok", "model": self.a.model, "model_ids": [self.a.model], "max_batch_size": self.a.parallel,
             "warming": False,
             "live": {"connections": self.active, "waiting": 0}}
        with self.lock:
            pc = self.cache_stats()
            if pc is not None:
                h["chats_continued"] = 0
                h["chats_continued_prefill_tokens"] = 0
                h["prompt_cache"] = pc
        return h

    def metrics(self):
        h = self.health()
        lines = []

        def put(name, v):
            lines.append(f"# TYPE tensorfold:{name} counter\ntensorfold:{name} {v}")
        if "prompt_cache" in h:
            for k, v in h["prompt_cache"].items():
                if not isinstance(v, list):
                    put("prompt_cache_" + k, v)
            for k in SPILL_FIELDS if isinstance(h["prompt_cache"].get("spill"), list) else ():
                lines.append(f"# TYPE tensorfold:spill_{k} counter")
                lines.extend(f'tensorfold:spill_{k}{{rank="{o["rank"]}"}} {o[k]}' for o in h["prompt_cache"]["spill"])
        return "\n".join(lines) + "\n"


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):          # no getfqdn: a reverse lookup can hang for seconds without DNS
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake: Fake = None

    def log_message(self, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def gone(self):
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            if r and self.connection.recv(1, socket.MSG_PEEK) == b"":
                raise Gone()
        except (OSError, ValueError):
            raise Gone()

    def do_GET(self):
        f = self.fake
        if self.path == "/health":
            self.send(200, f.health())
        elif self.path == "/metrics":
            self.send(200, f.metrics(), "text/plain")
        elif self.path == "/v1/models":
            self.send(200, {"object": "list", "data": [{"id": f.a.model, "object": "model", "owned_by": "tensorfold"}]})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        f = self.fake
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.send(400, {"error": {"message": "bad json"}})
        if self.path == "/fake/stop":
            os._exit(0)
        if self.path == "/tokenize":
            return self.send(200, {"count": len(f.tokens(body.get("messages") or [])), "max_model_len": f.window})
        if self.path != "/v1/chat/completions":
            return self.send(404, {"error": "not found"})
        messages, max_tokens = body.get("messages") or [], int(body.get("max_tokens") or 64)
        if body.get("stream"):
            return self.stream(messages, max_tokens, body)
        try:
            r = f.run(messages, max_tokens, self.gone)
        except ValueError as e:
            return self.send(400, {"error": {"message": str(e), "type": "invalid_request_error",
                                             "code": "context_length_exceeded"}})
        except Gone:
            self.close_connection = True
            return
        self.send(200, {"id": "fake", "object": "chat.completion", "model": f.a.model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                                     "finish_reason": r["finish"]}],
                        "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                                  "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
                                  "prompt_tokens_details": {"cached_tokens": r["cached"]}},
                        "tensorfold": r["tensorfold"]})

    def stream(self, messages, max_tokens, body):
        f = self.fake
        started = []

        def chunk(obj):
            data = b"data: " + json.dumps(obj).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def head():
            if not started:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                started.append(1)

        def emit(piece):
            head()
            try:
                chunk({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            except OSError:
                raise Gone()
        try:
            r = f.run(messages, max_tokens, self.gone, emit)
        except ValueError as e:
            if not started:
                return self.send(400, {"error": {"message": str(e)}})
            return
        except Gone:
            self.close_connection = True
            return
        try:
            head()
            chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": r["finish"]}]})
            chunk({"choices": [], "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                                            "prompt_tokens_details": {"cached_tokens": r["cached"]}},
                   "tensorfold": r["tensorfold"]})
            data = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(data), data))
            self.wfile.flush()
        except OSError:
            self.close_connection = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18888)
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--context", type=int, default=131072, help="the window one request may use")
    ap.add_argument("--grid", type=int, default=2048)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--mem-points", type=int, default=8)
    ap.add_argument("--spill-dir")
    ap.add_argument("--spill-dir-rank1")
    ap.add_argument("--spill-min-tokens", type=int, default=8192)
    ap.add_argument("--spill-min-free-gib", type=float, default=0)
    ap.add_argument("--disk-free-bytes", type=int, default=0, help="simulate a disk this big (less what each rank "
                                                                   "stores); 0: statvfs")
    ap.add_argument("--write-ms", type=float, default=40)
    ap.add_argument("--restore-ms-per-block", type=float, default=15)
    ap.add_argument("--serial-prefill", action="store_true", help="one prompt pass at a time (the Zig engine's admission)")
    ap.add_argument("--prefill-us-per-token", type=float, default=20)
    ap.add_argument("--decode-ms-per-token", type=float, default=2)
    ap.add_argument("--mutate", default="", choices=("", "restore-wrong", "no-disk", "ignore-crc", "conc-wrong",
                                                     "no-token-sha", "ignore-presence", "admit-over-window"))
    a = ap.parse_args()
    Handler.fake = Fake(a)
    srv = Server((a.host, a.port), Handler)
    print(f"fake_server: http://{a.host}:{a.port} spill={a.spill_dir} serial_prefill={a.serial_prefill} "
          f"mutate={a.mutate or 'none'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
