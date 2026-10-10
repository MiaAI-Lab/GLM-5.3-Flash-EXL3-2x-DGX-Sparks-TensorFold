#!/bin/bash
# spill_blocks_mutants.sh <image | dir> [names]   CPU only: tools/spill_blocks_check.py must pass on patch 0110 as it
# is and FAIL on each faithful mutant of its fixes (the code before the fix, or one of its parts taken out).
#   image: one scripts/prepare.sh built with this patch (each run in a fresh container, --network none);
#   dir:   a folder holding the patched tensorfold/ package, run with $PYTHON (default python3) on a copy of it.
# Each run applies one mutation (python escapes; an anchor that does not match exactly once is an ERROR, never a
# pass); a kill needs the check's non-zero exit AND a FAIL line. names: a bash regex, only the runs it matches.
# Exit 0 only when the unmutated check passes and every mutant is killed.
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
old = os.environ["OLD"].encode().decode("unicode_escape")
if old:
    p = os.path.join(os.environ["TREE"], os.environ["FILE"])
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"ANCHOR {n}")
    open(p, "w").write(s.replace(old, os.environ["NEW"].encode().decode("unicode_escape")))
'
bad=0
run() {   # name, file (under tensorfold/), old, new (name "patched": no mutation)
  [[ "$1" =~ $ONLY ]] || return 0
  if [ -d "$TARGET" ]; then
    rm -rf "$SCRATCH/t" && mkdir -p "$SCRATCH/t" && cp -r "$TARGET/tensorfold" "$SCRATCH/t/" &&
    find "$SCRATCH/t" -name __pycache__ -type d -prune -exec rm -rf {} + &&
    TREE="$SCRATCH/t/tensorfold" FILE="$2" OLD="$3" NEW="$4" "$PYTHON" -c "$MUTATE" > "$OUT" 2>&1 &&
    (cd "$SCRATCH" && PYTHONPATH="$SCRATCH/t" CUDA_VISIBLE_DEVICES= timeout 300 "$PYTHON" "$C") >> "$OUT" 2>&1
  else
    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 -v "$C":/c.py:ro \
      -e FILE="$2" -e OLD="$3" -e NEW="$4" -e MUTATE="$MUTATE" --entrypoint sh "$TARGET" -c '
      TREE=$(python3 -c "import os, tensorfold; print(os.path.dirname(tensorfold.__file__))") &&
      TREE=$TREE python3 -c "$MUTATE" && cd / && timeout 300 python3 /c.py' > "$OUT" 2>&1
  fi
  rc=$?
  if grep -q "^ANCHOR" "$OUT"; then echo "ERROR $1: $(grep ANCHOR "$OUT")"; bad=1
  elif [ "$1" = patched ]; then
    if [ $rc = 0 ] && grep -q "^all passed" "$OUT" && ! grep -q "^FAIL" "$OUT"; then
      echo "patched: all passed ($(grep -c '^ok' "$OUT") checks)"
    else echo "patched: FAILS, exit $rc ($(grep -m3 -E '^FAIL|Error' "$OUT" | tr '\n' ' '))"; bad=1; fi
  elif [ $rc != 0 ] && grep -q "^FAIL" "$OUT"; then
    echo "KILLED $1 ($(grep -c '^FAIL' "$OUT") FAIL lines, e.g. $(grep -m1 '^FAIL' "$OUT" | cut -c1-110))"
  elif [ $rc = 0 ]; then echo "SURVIVED $1"; bad=1
  else echo "CRASHED $1, exit $rc: $(grep -E 'Error|error' "$OUT" | tail -1)"; bad=1; fi
}
B=cuda/spill_blocks.py; MU=families/glm5_next/cuda/multi.py
run patched "" "" ""
# review 1: the tail block copied at keep
run tail-staged-later $B "        cut = [(h, _host(torch.cat(self._parts(extent, s, e)))) for s, e, h in new[-1:] if e - s < self.block]" "        cut = []"
run tail-copied-and-staged $B "            job = Job(pt, new[:len(new) - len(cut)], (items, layer)" "            job = Job(pt, new, (items, layer)"
run tail-not-queued $B "        for h, data in cut:                                 # (written before the blocks the pumps stage)\n            self.jobs.put((\"rows\", job, h, data))\n" ""
rm -f "$OUT"
echo "exit $bad"
exit $bad
