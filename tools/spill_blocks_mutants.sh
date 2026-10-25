#!/bin/bash
# spill_blocks_mutants.sh <dir> [names]   CPU only: tools/spill_blocks_check.py must pass on patch 0128 as it is and
# FAIL on each faithful mutant of its fixes (the code before the fix, or one of its parts taken out). <dir> holds the
# patched tensorfold/ package; each run works on a copy of it with $PYTHON (default python3). In the image
# scripts/prepare.sh built:
#   docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools:/t:ro" --entrypoint bash \
#     tensorfold-glm53:v0.6.0 /t/spill_blocks_mutants.sh /usr/local/lib/python3.12/dist-packages
# Each run applies one mutant (one or more edits, python escapes; an anchor that does not match exactly once is an
# ERROR, never a pass); a kill needs the check's non-zero exit AND a FAIL line. names: a bash regex, only the runs it
# matches. Exit 0 only when the unmutated check passes and every mutant is killed. Not mutated (no CPU check reaches
# them): the CUDA events that keep a read's pinned slot from being read into again before the engine's copy out of it
# ended (``_recycle``).
set -u
TARGET=$1; ONLY=${2:-.}
HERE=$(cd "$(dirname "$0")" && pwd)
C="$HERE/spill_blocks_check.py"
PYTHON=${PYTHON:-python3}
OUT=$(mktemp)
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH" "$OUT"' EXIT
MUTATE='
import os, sys
for i in range(int(os.environ["EDITS"])):
    old = os.environ[f"OLD{i}"].encode().decode("unicode_escape")
    p = os.path.join(os.environ["TREE"], os.environ[f"FILE{i}"])
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"ANCHOR {n} (edit {i})")
    open(p, "w").write(s.replace(old, os.environ[f"NEW{i}"].encode().decode("unicode_escape")))
'
bad=0
run() {   # name, then (file under tensorfold/, old, new) for each edit (name "patched": none)
  local name=$1 i=0
  shift
  [[ "$name" =~ $ONLY ]] || return 0
  local env=()
  while [ $# -ge 3 ]; do env+=("FILE$i=$1" "OLD$i=$2" "NEW$i=$3"); i=$((i + 1)); shift 3; done
  env+=("EDITS=$i")
  rm -rf "$SCRATCH/t" && mkdir -p "$SCRATCH/t" && cp -r "$TARGET/tensorfold" "$SCRATCH/t/" &&
  find "$SCRATCH/t" -name __pycache__ -type d -prune -exec rm -rf {} + &&
  env "${env[@]}" TREE="$SCRATCH/t/tensorfold" "$PYTHON" -c "$MUTATE" > "$OUT" 2>&1 &&
  (cd "$SCRATCH" && PYTHONPATH="$SCRATCH/t" CUDA_VISIBLE_DEVICES= TEST_TIMEOUT_S=${TEST_TIMEOUT_S:-120} timeout 300 "$PYTHON" "$C") \
    >> "$OUT" 2>&1
  rc=$?
  set -- "$name"
  if grep -q "^ANCHOR" "$OUT"; then echo "ERROR $1: $(grep ANCHOR "$OUT")"; bad=1
  elif [ "$1" = patched ]; then
    if [ $rc = 0 ] && grep -q "^all passed" "$OUT" && ! grep -q "^FAIL" "$OUT"; then
      echo "patched: all passed ($(grep -c '^ok' "$OUT") checks)"
    else echo "patched: FAILS, exit $rc ($(grep -m3 -E '^FAIL|Error' "$OUT" | tr '\n' ' '))"; bad=1; fi
  elif [ $rc != 0 ] && grep -q "^FAIL " "$OUT"; then
    echo "KILLED $1 ($(grep -c '^FAIL ' "$OUT") FAIL lines, e.g. $(grep -m1 '^FAIL ' "$OUT" | cut -c1-110))"
  elif [ $rc = 0 ]; then echo "SURVIVED $1"; bad=1
  else echo "CRASHED $1, exit $rc: $(grep -E 'Error|error' "$OUT" | tail -1)"; bad=1; fi
}
B=cuda/spill_blocks.py; MU=families/glm5_next/cuda/multi.py
run patched
# review 1: the tail block copied at keep
run tail-staged-later $B "        cut = [(h, _host(torch.cat(self._parts(extent, s, e)))) for s, e, h in new[-1:] if e - s < self.block]" "        cut = []"
run tail-copied-and-staged $B "            job = Job(pt, new[:len(new) - len(cut)], (items, layer)" "            job = Job(pt, new, (items, layer)"
run tail-not-queued $B "        for h, data in cut:                                 # (written before the blocks the pumps stage)\n            self.jobs.put((\"rows\", job, h, data))\n" ""
# review 2: a move of an extent being read waits for neither the read nor the disk
run move-waits-for-read $MU "        if keep is None:\n            return\n        h = getattr(x, \"loading\", None)\n        if h is not None and h.lj is not None and keep < h.n:\n            self.disk.stop_load(h.lj, \"its extent shrank under the read\")\n" "        h = getattr(x, \"loading\", None)\n        while h is not None and h.lj is not None and not all(it[2].done() for it in list(h.lj.queued.values())):\n            time.sleep(0.01)\n        if keep is None:\n            return\n"
run read-at-first-base $B "        self.point, self.where = point, where\n" "        self.point, self.where = point, where\n        self.base0 = where.base\n" \
    $B "        base = lj.where.base\n" "        base = lj.base0\n"
# the reader reads, the engine copies between rounds; LOADPART takes rank 1 as far
run reader-copies $B "            fut = self._readers.submit(self._read_block, h, pt.crcs[i], self.read_slots[b], nb)\n" "            fut = self._readers.submit(self._read_block, h, pt.crcs[i], self.read_slots[b], nb)\n            fut.result()\n            self._copy_in(lj, self.read_slots[b], s, e)\n            self.stats[\"loaded_bytes\"] += nb\n"
run no-step $MU "                if self.disk.step(h.lj, budget=left):\n" "                if False:\n"
# review (wojo#2): one budget a pause for every read in flight, and each read first in turn
run budget-each-read $MU "                left = deadline - time.perf_counter()\n" "                left = budget\n"
run ready-before-copies $MU "        if lj.error is not None or lj.next >= lj.total:\n            return True\n" "        return True\n"
run budget-same-first $MU "            k = self.load_turn % len(live)" "            k = 0"
run loadpart-not-sent $MU "                    self._emit(LOADPART, [h.lid, h.lj.next])\n" "                    pass\n"
run loadpart-not-applied $MU "            elif op == LOADPART:\n                self._load_part(*p)" "            elif op == LOADPART:\n                pass"
run loadpart-past-upto $B "        end = lj.total if upto is None else min(int(upto), lj.total)\n" "        end = lj.total\n"
run loaded-without-rest $B "        if lj.error is None:\n            self.step(lj, upto=lj.total)\n" ""
# review 3, shared flag 1 and 2: a restore keeps rank 0's shared-prefix flag on every rank
run restore-unshared $MU "        snap.shared = bool(shared)" "        snap.shared = False" \
    $MU "        self._emit(LOADED, [h.lid, shared])" "        self._emit(LOADED, [h.lid])"
run each-rank-own-flag $MU "        snap.shared = bool(shared)" "        snap.shared = bool(pt.shared)"
run loaded-without-flag $MU "        self._emit(LOADED, [h.lid, shared])" "        self._emit(LOADED, [h.lid])"
# shared flag 3: ids kept again as shared make their point shared, durably
run no-upgrade $B "                pt = self.index.get(key)\n                if pt is not None:\n                    pt.used = time.time()\n                elif key in self.waiting:" "                pt = None\n                if key in self.index:\n                    self.index[key].used = time.time()\n                elif False:"
run upgrade-not-durable $B "                    if key in self.index:                   # its json again, on the writer (\`\`_mark_shared\`\`)\n                        self.jobs.put((\"shared\", None, key, None))\n" ""
run waiting-not-flagged $B "                elif key in self.waiting:                   # still being written: its json takes the flag then\n                    pt = self.waiting[key].point\n" ""
run written-no-rewrite $B "                if pt.shared and not shared:\n                    self.jobs.put((\"shared\", None, pt.key, None))\n" ""
run rewrite-resurrects $B "        if gone:\n            try:\n                os.remove(path)" "        if False:\n            try:\n                os.remove(path)"
run rewrite-not-put $B "        _put(path, json.dumps(meta).encode(), self.owner)\n" "        open(path, \"wb\").write(json.dumps(meta).encode())\n"
run rewrite-no-dir-fsync $B "        _put(path, json.dumps(meta).encode(), self.owner)\n        _fsync_dir(self.points_dir)\n" "        _put(path, json.dumps(meta).encode(), self.owner)\n"
run drain-skips-rewrites $B "        synced = threading.Event()                          # (behind every rewrite those jobs or a keep queued)\n        self.jobs.put((\"sync\", None, None, synced))\n        return synced.wait(None if end is None else max(0.0, end - time.monotonic()))\n" "        return True\n"
# review 4: a missing name is not damage to the names linked to its bytes
run lost-is-damage $B "                if isinstance(exc, FileNotFoundError):     # this name is gone: its links' bytes are not in question\n                    lj.lost.add(h)\n                elif isinstance(exc, (Corrupt, EOFError)):  # damaged bytes, not a passing I/O error\n" "                if isinstance(exc, (Corrupt, FileNotFoundError, EOFError)):   # damage, not a passing I/O error\n"
run lost-names-kept $B "            gone = bad | set(lost)\n" "            gone = bad\n"
run damage-not-widened $B "                if b in self.have and self.have[b][1] in self.inodes:\n                    bad |= self.inodes[self.have[b][1]][1]\n" "                pass\n"
run finish-load-drops-lost $MU "            self.disk.forget(h.key, lj.bad if lj is not None else (), lj.lost if lj is not None else ())" "            self.disk.forget(h.key, lj.bad if lj is not None else ())"
# review 5: no op without a sender; the prefix wait's counters in /metrics
run ops-unsent $MU "LOAD, FLUSH, LOADED, \\\\\n    HAS, LOADPART = range(1, 18)" "LOAD, FLUSH, SPILL, LOADED, \\\\\n    TOUCH, HAS, LOADPART = range(1, 20)" \
    $MU "       FLUSH: \"FLUSH\", LOADED: \"LOADED\", HAS: \"HAS\", LOADPART: \"LOADPART\"}" "       FLUSH: \"FLUSH\", SPILL: \"SPILL\", LOADED: \"LOADED\", TOUCH: \"TOUCH\", HAS: \"HAS\", LOADPART: \"LOADPART\"}"
run prefix-not-exported server/metrics.py "        if field in body:\n            body.setdefault(name, body[field])\n" "        pass\n"
# the page cache: every point file's pages dropped after its fsync (already so in the patch; pinned here)
run small-files-keep-pages $B "            os.fsync(fd)\n            _drop_cache(fd)                                 # (clean now" "            os.fsync(fd)\n            pass                                 # (clean now"
run pages-dropped-before-fsync $B "            os.fsync(fd)\n            _drop_cache(fd)                                 # (clean now" "            _drop_cache(fd)\n            os.fsync(fd)                                 # (clean now"
# a read slot turns every round: the round settles the reads once its verify pass is queued
run round-no-settle $MU "        if self.disk is not None and self.loads:\n            try:\n                self.disk.settle()" "        if False:\n            try:\n                self.disk.settle()"
run round-settle-always $MU "        if self.disk is not None and self.loads:\n            try:\n                self.disk.settle()" "        if self.disk is not None:\n            try:\n                self.disk.settle()"
run settle-oldest-only $B "        last = next((ev for _, ev in reversed(evs) if ev is not None), None)" "        last = next((ev for _, ev in evs if ev is not None), None)"
run settle-frees-none $B "        for b, _ in evs:\n            self.free_reads.put(b)\n        self.read_events = []\n" "        pass\n"
run settle-before-slots $B "        evs = getattr(self, \"read_events\", None)" "        evs = self.read_events"
# the writer: a write slot's event made at start and reused (wojo#2's pump made one a block, the writer dropped it)
run write-event-per-block $B "                    ev = self.slot_events[slot] if self.slot_events else None   # (made at start: a GPU's only)\n                    if ev is not None:\n                        ev.record()\n" "                    ev = None\n                    if self.dev_slot.device.type == \"cuda\":\n                        ev = torch.cuda.Event()\n                        ev.record()\n"
# LOADED on a rank with more blocks left than read slots: the wait frees the slots its copies hold
run hang-no-recycle-in-wait $B "                    if self.read_events:                    # the slots this step's own copies used: back to the reader,\n                        self._recycle(wait=True)            # or it waits for one and block i is never read\n" ""
# the read-ahead: 16 blocks by default (ten in the preallocated buffers), those buffers kept, the rest in power-of-two pieces
run read-ahead-default-9 $B "READ_BLOCKS = 16               #" "READ_BLOCKS = 9                #"
run read-ahead-default-32 $B "READ_BLOCKS = 16               #" "READ_BLOCKS = 32               #"
run read-ahead-default-17 $B "READ_BLOCKS = 16               #" "READ_BLOCKS = 17               #"
run read-ahead-replaces-preallocate $B "    bufs = _sp._PINNED.setdefault(\"read\", [])\n    want = read_blocks() * per - sum(b.numel() for b in bufs)\n" "    bufs = _sp._PINNED[\"read\"] = []\n    want = read_blocks() * per\n"
run read-ahead-charges-preallocate $B "    want = read_blocks() * per - sum(b.numel() for b in bufs)\n" "    want = read_blocks() * per\n"
run read-ahead-one-piece $B "        size = 1 << (max(min(want, _sp.CHUNK), per) - 1).bit_length()\n" "        size = max(want, per)\n"
run read-ahead-per-slot $B "        size = 1 << (max(min(want, _sp.CHUNK), per) - 1).bit_length()\n" "        size = per\n"
# the room (review 2026-10-24): the eviction order, what goes with a point, the quota, the turns, the floor, a restart
run victims-forks-first $B "        out = [(k, \"pruned_turns\") for k in sorted(turns, key=by_used)]\n        out += [(k, \"pruned_forks\") for k in sorted(forks, key=by_used)]\n" "        out = [(k, \"pruned_forks\") for k in sorted(forks, key=by_used)]\n        out += [(k, \"pruned_turns\") for k in sorted(turns, key=by_used)]\n"
run victims-shared-early $B "        for _, keys, _, path in sorted(whole, key=lambda g: g[0]):\n" "        out += [(k, \"pruned_shared\") for k, c in conv.items() if c is None]\n        for _, keys, _, path in sorted(whole, key=lambda g: g[0]):\n"
run remove-keeps-files $B "        self.small_on_disk -= pt.small_bytes\n        for suffix in (\".json\", \".st\", \".ids\"):\n            try:\n                os.remove(self._point_path(key, suffix))\n" "        self.small_on_disk -= pt.small_bytes\n        for suffix in ():\n            try:\n                os.remove(self._point_path(key, suffix))\n"
run remove-keeps-blocks $B "        self.stats[why] += 1\n        self._gc(pt.blocks)\n" "        self.stats[why] += 1\n"
run room-takes-read $B "                if key in protect or key not in self.index or key in self.reading:\n" "                if key in protect or key not in self.index:\n"
run room-takes-pinned $B "            if b in self.have and b not in self.refs and b not in self.inflight and b not in self.pinned:\n" "            if b in self.have and b not in self.refs and b not in self.inflight:\n"
run no-trim $B "        for k in path[self.turns:]:\n" "        for k in path[len(path):]:\n"
run trim-reading $B "            if k not in self.reading and k not in self.inflight:\n                self._remove(k, \"trimmed_turns\")\n" "            if k not in self.inflight:\n                self._remove(k, \"trimmed_turns\")\n"
run trim-shared $B "        while cur is not None and cur in self.index and not self.index[cur].shared:\n" "        while cur is not None and cur in self.index:\n"
run no-quota $B "        if self.quota > 0:\n            self._trim_own(pt, need)\n" ""
run quota-takes-others $B "            for key, why in self._victims(only=c):\n" "            for key, why in self._victims():\n"
run floor-evicts-for-nothing $B "        if max(short, over) > self.stored():                # (the disk is full of other things): nothing to evict for\n" "        if False:\n"
run floor-ignored $B "        short = self.floor + need + self.backlog - self.disk_free()\n        over =" "        short = 0\n        over ="
run scan-keeps-temporary $B "                if f.startswith(\".\"):                       # a write cut off before its rename\n" "                if False:\n"
run scan-keeps-orphans $B "        for b in present - self.have.keys():\n" "        for b in ():\n"
run scan-keeps-incomplete-files $B "        for f in os.listdir(self.points_dir):\n            if f.split(\".\")[0] not in self.index:\n" "        for f in ():\n            if f.split(\".\")[0] not in self.index:\n"
run scan-no-ids-crc $B "                if zlib.crc32(raw) != m[\"ids_crc\"] or not os.path.exists(self._point_path(key, \".st\")):\n" "                if not os.path.exists(self._point_path(key, \".st\")):\n"
run prune-other-ranks $B "            mine = os.path.join(d, f\"rank{self.rank}\")\n            if os.path.isdir(mine):\n                shutil.rmtree(mine, ignore_errors=True)\n" "            if os.path.isdir(d):\n                shutil.rmtree(d, ignore_errors=True)\n"
run failed-put-leaves-tmp $B "    except BaseException:\n        try:\n            os.remove(tmp)\n        except OSError:\n            pass\n        raise\n    return crc\n" "    except BaseException:\n        raise\n    return crc\n"
# hardening: temporary files made new, never through a symlink; one engine a folder
run tmp-truncates $B "_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW\n" "_NEW = os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n" \
    $B "    except FileExistsError:\n        os.unlink(tmp)\n" "    except FileExistsError:\n        pass\n"
run tmp-no-excl $B "_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW\n" "_NEW = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW\n"
run no-lock cuda/spill.py "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n" "        pass\n"
run lock-after-prune $B "            self._held = _sp.hold(cfg.root, self.rank)    # (before any file of it is read, pruned or written)\n" "" \
    $B "            pruned_builds = self._prune_builds()\n            self._scan()\n        except _sp.Busy" "            pruned_builds = self._prune_builds()\n            self._scan()\n            self._held = _sp.hold(cfg.root, self.rank)\n        except _sp.Busy"
run busy-stops-engine $MU "        except Busy as exc:                     # (raised on every rank alike: the tier is off on all of them)\n            print(f\"[tensorfold] spill rank {self.rank}: {exc}: the spill tier is off" "        except KeyError as exc:\n            print(f\"[tensorfold] spill rank {self.rank}: {exc}: the spill tier is off"
run one-stream-unlocked cuda/spill.py "            self._held = hold(cfg.root, self.rank)\n" ""
# release: its own cap on the rows it copies to host memory
run release-uncapped $B "                fits = self.host_bytes + need <= self.release_cap\n" "                fits = True\n"
run release-not-counted $B "                if fits:\n                    self.host_bytes += need\n" ""
run release-never-let-go $B "                    with self.lock:\n                        self.host_bytes -= item.numel() * item.element_size()\n" "                    pass\n"
run release-cap-backlog $B "    return reserve // RELEASE_SHARE\n" "    return BACKLOG\n"
run cut-not-counted $B "            self.host_bytes += sum(t.numel() * t.element_size() for _, t in cut)   # (at most a block a state)\n" ""
# the prefix wait: off by default
run prefix-wait-default-on $MU "PREFIX_WAIT = os.environ.get(\"TF_GLM_PREFIX_WAIT\", \"0\") not in (\"\", \"0\")" "PREFIX_WAIT = os.environ.get(\"TF_GLM_PREFIX_WAIT\", \"1\") != \"0\""
run prefix-wait-ignores-flag $MU "        if not PREFIX_WAIT or not s.draft" "        if not s.draft"
run prefix-wait-no-minimum $MU "                if p <= m and p < len(prompt) and p - have >= self.g.shared and" "                if p <= m and p < len(prompt) and"
run prefix-never-ready $MU "        return lane is None or lane.decoding or lane.s.done or lane.st.pos >= p\n" "        return lane is None or lane.s.done\n"
run prefix-wait-blocks-queue $MU "                    stream.prefix_waited = True\n                    self.prefix.append((w, item))\n                    continue\n" "                    stream.prefix_waited = True\n                    self.prefix.append((w, item))\n                    break\n"
rm -f "$OUT"
echo "exit $bad"
exit $bad
