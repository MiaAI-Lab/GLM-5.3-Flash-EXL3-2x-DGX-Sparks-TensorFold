#!/bin/bash
# kv_shard_mutants.sh <dir> [names]   CPU only: tools/kv_shard_check.py must pass on the tree as it is and FAIL on each
# faithful mutant of the cross-part carry (TF_GLM_KV_SHARD_CARRY): the code with one of its guards, rules or steps taken
# out or turned the wrong way. <dir> holds the patched tensorfold/ package; each run works on a copy of it with $PYTHON
# (default python3). In the image scripts/prepare.sh built:
#   docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools:/t:ro" --entrypoint bash \
#     tensorfold-glm53:v0.6.0 /t/kv_shard_mutants.sh /usr/local/lib/python3.12/dist-packages
# Each run applies one mutant (one or more edits, python escapes; an anchor that does not match exactly once is an
# ERROR, never a pass); a kill needs the check's non-zero exit AND a FAIL line. names: a bash regex, only the runs it
# matches. Exit 0 only when the unmutated check passes and every mutant is killed. Not mutated (no CPU check reaches
# them): the NCCL calls themselves, the CUDA events that time a carry, and the collectives' order across real ranks
# (the check simulates the ranks one after the other in one process: the sender first, a receive with no send before
# it raises).
set -u
TARGET=$1; ONLY=${2:-.}
HERE=$(cd "$(dirname "$0")" && pwd)
C="$HERE/kv_shard_check.py"
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
  (cd "$SCRATCH" && PYTHONPATH="$SCRATCH/t" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= timeout 300 "$PYTHON" "$C") >> "$OUT" 2>&1
  rc=$?
  set -- "$name"
  if grep -q "^ANCHOR" "$OUT"; then echo "ERROR $1: $(grep ANCHOR "$OUT")"; bad=1
  elif [ "$1" = patched ]; then
    if [ $rc = 0 ] && grep -q "^ALL PASS" "$OUT" && ! grep -q "^FAIL" "$OUT"; then
      echo "patched: all passed ($(grep -c '^PASS' "$OUT") checks)"
    else echo "patched: FAILS, exit $rc ($(grep -m3 -E '^FAIL|Error' "$OUT" | tr '\n' ' '))"; bad=1; fi
  elif [ $rc != 0 ] && grep -q "^FAIL " "$OUT"; then
    echo "KILLED $1 ($(grep -c '^FAIL ' "$OUT") FAIL lines, e.g. $(grep -m1 '^FAIL ' "$OUT" | cut -c1-110))"
  elif [ $rc = 0 ]; then echo "SURVIVED $1"; bad=1
  else echo "CRASHED $1, exit $rc: $(grep -E 'Error|error' "$OUT" | tail -1)"; bad=1; fi
}
MU=families/glm5_next/cuda/multi.py; PO=families/glm5_next/cuda/pool.py; SH=families/glm5_next/cuda/shard.py; SB=cuda/spill_blocks.py; EN=families/glm5_next/cuda/engine.py
run patched
# the collective: the other ranks hold the op before rank 0 sends (ADMIT copy 2, XMOVE)
run drop-sync $MU "            if copy == 2:                     # a carry is a collective: the other ranks hold this ADMIT before rank 0 sends\n                self._sync()\n" ""
run xmove-no-sync $MU "        self._sync()                           # a carry is a collective: rank 1 holds the op before rank 0 sends\n" ""
run follower-copies $MU "                self._carry(hit.extent.base, x.base, cut, \"admit\")" "                self.arena.copy(hit.extent.base, x.base, cut)"
# the transport: the rows, byte for byte, between the two owners only
run carry-cut-minus-1 $PO "shard.send_rows(nc, do, _byte_rows(p, shard.local(src), n), p.spanned)" "shard.send_rows(nc, do, _byte_rows(p, shard.local(src), n - 1), p.spanned)" \
    $PO "shard.recv_rows(nc, so, _byte_rows(p, shard.local(dst), n), p.spanned)" "shard.recv_rows(nc, so, _byte_rows(p, shard.local(dst), n - 1), p.spanned)"
run skip-replicated-copy $PO "                _move(p.tensor, a, b, min(-(-n // p.div), p.tensor.shape[0] - max(a, b)), p.spanned)\n            elif relay:" "                pass\n            elif relay:"
run bystander-receives $PO "            elif shard.RANK == do:\n                shard.recv_rows" "            else:\n                shard.recv_rows"
run span-send-by-memcpy $SH "        put_rows(buf[:k], rows[i:i + k], spanned)" "        put_rows(buf[:k], rows[i:i + k], False)"
run span-recv-by-memcpy $SH "        put_rows(rows[i:i + k], buf[:k], spanned)" "        put_rows(rows[i:i + k], buf[:k], False)"
run pool-move-crosses-by-default $PO "size: int | None = None, cross: bool = False) -> int:" "size: int | None = None, cross: bool = True) -> int:"
run xmove-not-applied $MU "            elif op == XMOVE:\n                self._xmove(self.pool.get(p[0]), p[1], p[2], p[3])" "            elif op == XMOVE:\n                pass"
run grow-fits-true $MU "            if hi - lo - held >= size:\n                return True\n        return False\n" "            return True\n        return False\n"
run grow-fits-held-flipped $MU "and id(y) not in gone and (y.owner is not None or not y.kept))" "and id(y) not in gone and (y.owner is not None or y.kept))"
# the pressure rule: a carry only to avoid a real cost
run pressure-count-balance $MU "        if self._fits_unevicted(hit, need, others):\n            return None\n" "        if self._fits_unevicted(hit, need, others) and self.pool.live(self.pool.half_of(hit.extent)) - min(self.pool.live(q) for q in range(self.pool.parts)) < 2:\n            return None\n"
run pressure-without-eviction-condition $MU "        if self._fits_unevicted(hit, need, others):\n            return None\n" ""
run pressure-no-headroom $MU "            want = need + ALIGN * (pool.live(q) + 1)\n            if pool.place(want, half=q)" "            want = need\n            if pool.place(want, half=q)"
run pressure-may-evict-in-target $MU "            if pool.place(want, half=q) is not None or pool.compactable(q) >= want:\n                base = self._room_free(want, q)" "            if True:\n                base = self._room(want, [hit.extent], half=q)"
run unevicted-ignores-lone-others $MU "        if others:\n            return not self.tune.lone and pool.place(need, half=p) is not None" "        if others:\n            return pool.place(need, half=p) is not None"
run unevicted-compacts-for-others $MU "            return not self.tune.lone and pool.place(need, half=p) is not None" "            return not self.tune.lone"
run small-ignored-pressure $MU "            return None\n        if len(hit.ids) < CARRY_MIN:" "            return None\n        if False:"
run small-uses-need $MU "            return None\n        if len(hit.ids) < CARRY_MIN:" "            return None\n        if need < CARRY_MIN:"
run small-ignored-others $MU "        pool = self.pool\n        if len(hit.ids) < CARRY_MIN:" "        pool = self.pool\n        if False:"
run small-ignored-rescue $MU "room either.\"\"\"\n\n        if len(hit.ids) < CARRY_MIN:" "room either.\"\"\"\n\n        if False:"
run recent-copy-1-untouched $MU "        if copy == 2 and getattr(hit, \"own\", False):" "        if copy and getattr(hit, \"own\", False):"
# counters
run no-part-never-restored $MU "self.no_part = spent if got is not None or hit.extent.owner is None else refused" "self.no_part = refused"
run no-part-always-spent $MU "self.no_part = spent if got is not None or hit.extent.owner is None else refused" "self.no_part = spent"
run counted-at-decision $MU "        self.carry_kind = \"rescued\"\n        return base, 2" "        self.carry[\"rescued\"] += 1\n        return base, 2"
run never-counted $MU "        if copy == 2 and self.carry_kind:         # (counted once the ADMIT is queued: a placement that failed earlier is not)\n            self.carry[self.carry_kind] += 1\n" ""
# the XMOVE rows are rank 0's
run follower-recomputes-used $MU "        self._carry(old, base, used, \"live\")" "        self._carry(old, base, mine, \"live\")"
run xmove-used-with-pending $MU "        used = self._used(x, pending=False)\n        if used > size:\n            return False\n" "        used = self._used(x)\n"
run xmove-follower-refuses $MU "        if used > size:                     # (rank 0's own arguments: checked before anything is sent)\n" "        if self._used(x) > used or used > size:\n"
# live streams: no bouncing
run no-headroom-in-target $MU "            want = size + ALIGN * (pool.live(q) + 1)\n            if pool.place(want, half=q)" "            want = size\n            if pool.place(want, half=q)" \
    $MU "        base = self._room_free(size + ALIGN * (pool.live(q) + 1), q)" "        base = self._room_free(size, q)"
run no-cooldown $MU "and self.xticks - lane.moved_at >= RELOCATE_HOLD)" "and True)"
run move-back-allowed $MU "(lane.came_from == q and self.xticks - lane.moved_at < RELOCATE_HOLD)" "False"
run no-entered-guard $MU "        if self.xticks - self.entered.get(p, -(1 << 30)) >= RELOCATE_HOLD:" "        if True:"
run no-cheapest-stream $MU "            for m in self.lanes.values():\n                if m is lane or" "            for m in []:\n                if m is lane or"
run hold-1 $MU "RELOCATE_HOLD = 256\n" "RELOCATE_HOLD = 1\n"
# the settings
run lone-not-forced-off $MU "    if not shard.ON or not tune.lone or not KV_CARRY:   # (TF_GLM_KV_SHARD_CARRY=0: the decoder as before the carry)\n        return tune\n" "    return tune\n"
run lone-off-at-carry0 $MU "    if not shard.ON or not tune.lone or not KV_CARRY:" "    if not shard.ON or not tune.lone:"
run relocate-second-carry $MU "                lane.xwhy = None\n                return True\n            break\n" "                lane.xwhy = None\n                return True\n"
run init-ignores-shard-tune $MU "self.tune = shard_tune(tune if tune is not None else MultiSettings.from_env())" "self.tune = tune if tune is not None else MultiSettings.from_env()"
# the time of a carry
run ms-own-side-only $MU "        if self.w.world > 2:\n            ms = max(" "        if False:\n            ms = max("
run ms-gathered-at-two-ranks $MU "        if self.w.world > 2:\n            ms = max(" "        if self.w.world > 1:\n            ms = max("
# TP3 warm-up
run warm-not-called $EN "            shard.warm(self.comm, w.device)         # (TP3 and up: the pairs' send/recv connections now, not at a carry)\n" ""
run warm-at-two-ranks $SH "    if WORLD < 3 or not carry_on() or relayed(nc):" "    if WORLD < 2 or not carry_on() or relayed(nc):"
run warm-pairs-reversed $SH "        for b in range(a + 1, WORLD):" "        for b in range(WORLD - 1, a, -1):"
run warm-both-send-first $SH "            elif RANK == b:\n                _recv(nc, got, a, stream)\n                _send(nc, out, a, stream)" "            elif RANK == b:\n                _send(nc, out, a, stream)\n                _recv(nc, got, a, stream)"
run import-per-piece $SH "    w = rows.view(torch.int32)\n    nc._check(nc.lib.ncclSend(" "    from tensorfold.cuda.comm import _DTYPES\n\n    w = rows.view(torch.int32)\n    nc._check(nc.lib.ncclSend("
# what is refused: rows that something pins
run pins-all-gone $MU "        if getattr(x, \"loading\", None) is not None:\n            return \"loading\"\n        if self._pinned()(x):\n            return \"pinned\"\n        lane = self.lanes.get(x.owner) if x.owner is not None else None\n        if x.twin or (lane is not None and lane.twin is not None):\n            return \"twin\"\n        if lane is not None and self.partial is not None and self.partial.contains(lane.sid):\n            return \"partial\"\n        return None" "        return None"
run pins-no-loading $MU "        if getattr(x, \"loading\", None) is not None:\n            return \"loading\"\n        if self._pinned()(x):" "        if self._pinned()(x):"
run pins-no-pinned $MU "        if self._pinned()(x):\n            return \"pinned\"\n" ""
run pins-no-twin $MU "        if x.twin or (lane is not None and lane.twin is not None):\n            return \"twin\"\n" ""
run pins-no-partial $MU "        if lane is not None and self.partial is not None and self.partial.contains(lane.sid):\n            return \"partial\"\n        return None" "        return None"
# the balance rule
run min-0 $MU "CARRY_MIN = 2048\n" "CARRY_MIN = 0\n"
# the rescue and the others branch
run rescue-off $MU "                got = self._rescue(hit, need, why) if sharded else None" "                got = None"
run others-off $MU "                if base is None and sharded:             # its part is full: free rows of another carry them instead" "                if False:"
# the LRU
run touch-carried-own $MU "        if copy == 2 and getattr(hit, \"own\", False):\n            return\n        self._touch(hit)" "        self._touch(hit)"
# live streams
run relocate-not-called $MU "            elif KV_CARRY and self.pool.half:\n                ok = self._relocate(l)" "            elif False:\n                ok = self._relocate(l)"
run xmove-no-rebind $MU "        self._carry(old, base, used, \"live\")\n        if x.owner is not None:\n            self.lanes[x.owner].st.bind(x.base, x.size)\n" "        self._carry(old, base, used, \"live\")\n"
run refusal-recounted $MU "        if lane.xwhy != why:\n            self._refused(why)" "        if True:\n            self._refused(why)"
# the spill tier
run no-spill-salt $MU "        return in_part(c.lineage, self.pool.half_of(c.extent)) if self.pool.half else c.lineage" "        return c.lineage"
run salt-unsharded $MU "        return in_part(c.lineage, self.pool.half_of(c.extent)) if self.pool.half else c.lineage" "        return in_part(c.lineage, self.pool.half_of(c.extent))"
run salt-stacks $SB "    return f\"{head.partition('#')[0]}#{int(part)}{bar}{rest}\"" "    return f\"{head}#{int(part)}{bar}{rest}\""
run no-release-on-cross-move $MU "            self._release_rows(x, 0 if cross else None)" "            self._release_rows(x, None)"
rm -f "$OUT"
echo "exit $bad"
exit $bad
