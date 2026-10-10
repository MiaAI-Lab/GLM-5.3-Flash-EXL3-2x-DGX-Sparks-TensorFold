#!/bin/bash
# spill_blocks_mutants.sh <dir> [names]   CPU only: tools/spill_blocks_check.py must pass on patch 0110 as it is and
# FAIL on each faithful mutant of its fixes (the code before the fix, or one of its parts taken out). <dir> holds the
# patched tensorfold/ package; each run works on a copy of it with $PYTHON (default python3). In the image
# scripts/prepare.sh built:
#   docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools:/t:ro" --entrypoint bash \
#     tensorfold-glm53:v0.6.0 /t/spill_blocks_mutants.sh /usr/local/lib/python3.12/dist-packages
# Each run applies one mutant (one or more edits, python escapes; an anchor that does not match exactly once is an
# ERROR, never a pass); a kill needs the check's non-zero exit AND a FAIL line. names: a bash regex, only the runs it
# matches. Exit 0 only when the unmutated check passes and every mutant is killed. Not mutated (no CPU check reaches
# them): the CUDA events that order a read's copies around a move (``moving``'s wait on ``last``, the reader's wait on
# ``moved``).
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
  (cd "$SCRATCH" && PYTHONPATH="$SCRATCH/t" CUDA_VISIBLE_DEVICES= timeout 300 "$PYTHON" "$C") >> "$OUT" 2>&1
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
run move-waits-for-read $MU "        if keep is None:\n            return\n        h = getattr(x, \"loading\", None)\n        if h is not None and h.lj is not None:\n            h.lj.done.wait()\n        if self.disk is not None:\n" "        h = getattr(x, \"loading\", None)\n        if h is not None and h.lj is not None:\n            h.lj.done.wait()\n        if self.disk is not None and keep is not None:\n"
run read-at-first-base $B "        self.point, self.where = point, where\n" "        self.point, self.where = point, where\n        self.base0 = where.base\n" \
    $B "                base = lj.where.base\n" "                base = lj.base0\n"
run move-not-held $MU "        with self.disk.moving(h.lj) if h is not None and h.lj is not None else contextlib.nullcontext():" "        with contextlib.nullcontext():"
run moving-without-lock $B "        with lj.lock:\n            if lj.last is not None:" "        with contextlib.nullcontext():\n            if lj.last is not None:"
run copies-without-lock $B "            with lj.lock:                                   # (the extent stays" "            with contextlib.nullcontext():                                   # (the extent stays"
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
run ops-unsent $MU "LOAD, FLUSH, LOADED, \\\\\n    HAS = range(1, 17)" "LOAD, FLUSH, SPILL, LOADED, \\\\\n    TOUCH, HAS = range(1, 19)" \
    $MU "       FLUSH: \"FLUSH\", LOADED: \"LOADED\", HAS: \"HAS\"}" "       FLUSH: \"FLUSH\", SPILL: \"SPILL\", LOADED: \"LOADED\", TOUCH: \"TOUCH\", HAS: \"HAS\"}"
run prefix-not-exported server/metrics.py "        if field in body:\n            body.setdefault(name, body[field])\n" "        pass\n"
# the page cache: every point file's pages dropped after its fsync (already so in the patch; pinned here)
run small-files-keep-pages $B "            os.fsync(fd)\n            _drop_cache(fd)                                 # (clean now" "            os.fsync(fd)\n            pass                                 # (clean now"
run pages-dropped-before-fsync $B "            os.fsync(fd)\n            _drop_cache(fd)                                 # (clean now" "            _drop_cache(fd)\n            os.fsync(fd)                                 # (clean now"
rm -f "$OUT"
echo "exit $bad"
exit $bad
