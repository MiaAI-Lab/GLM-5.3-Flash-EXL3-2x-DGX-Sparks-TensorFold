#!/bin/bash
# mutants.sh <0109 patch> <kv_tier_check.py> [image] [names]   CPU only: the PR's check must pass on the patched code,
# fail on the stock code, and FAIL on each mutant of the patched code. Each run applies the patch with the recipe's own
# `patch -p0` in the image's site-packages, then the mutation (python escapes; an anchor that does not match exactly
# once is an ERROR, never a pass); a kill needs the check's non-zero exit AND a FAIL line (a hang prints one after
# TEST_TIMEOUT_S). image: v1.10's (tensorfold-glm53:v0.6.0-a1897d59, recipe 549fdc5). names: a bash regex, only the
# runs it matches (default all). On a machine shared with another serve each run waits for >= $MIN_GB GB available and
# is capped at 24 GB. Exit 0 only when the patched check passes, the stock code fails it and every mutant is killed.
set -u
P=$(readlink -f "$1"); C=$(readlink -f "$2"); IMAGE=${3:-tensorfold-glm53:v0.6.0-a1897d59}; ONLY=${4:-.}
MIN_GB=${MIN_GB:-30}
OUT=$(mktemp)
bad=0
run() {   # name, file, old, new (name "patched": the patch alone; "stock": no patch)
  [[ "$1" =~ $ONLY ]] || return 0
  while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt "$MIN_GB" ]; do sleep 10; done
  docker run --rm --network none --memory 24g -e CUDA_VISIBLE_DEVICES= -e TEST_TIMEOUT_S=120 -e PYTHONUNBUFFERED=1 \
    -v "$P":/x.patch:ro -v "$C":/c.py:ro -e NAME="$1" -e FILE="$2" -e OLD="$3" -e NEW="$4" --entrypoint sh "$IMAGE" -c '
    cd /usr/local/lib/python3.12/dist-packages && { [ "$NAME" = stock ] || patch -s -p0 < /x.patch; } &&
    python3 - <<EOF
import os, sys
old = os.environ["OLD"].encode().decode("unicode_escape")
if old:
    p = "tensorfold/families/glm5_next/cuda/" + os.environ["FILE"]
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"ANCHOR {n}")
    open(p, "w").write(s.replace(old, os.environ["NEW"].encode().decode("unicode_escape")))
EOF
    cd / && timeout 600 python3 /c.py' > "$OUT" 2>&1
  rc=$?
  if grep -q "^ANCHOR" "$OUT"; then echo "ERROR $1: $(grep ANCHOR "$OUT")"; bad=1
  elif [ "$1" = patched ]; then
    if [ $rc = 0 ] && grep -q "^ALL PASS" "$OUT" && ! grep -q "^FAIL" "$OUT"; then
      echo "patched: ALL PASS ($(grep -c '^PASS' "$OUT") checks)"
    else echo "patched: FAILS, exit $rc ($(grep -m3 -E '^FAIL|Error' "$OUT" | tr '\n' ' '))"; bad=1; fi
  elif [ $rc != 0 ] && grep -q "^FAIL" "$OUT"; then
    echo "KILLED $1 ($(grep -c '^FAIL' "$OUT") FAIL lines, e.g. $(grep -m1 '^FAIL' "$OUT" | cut -c1-120))"
  elif [ $rc = 0 ]; then echo "SURVIVED $1"; bad=1
  else echo "CRASHED $1, exit $rc: $(grep -E 'Error|error' "$OUT" | tail -1)"; bad=1; fi
}
K=kvtier.py; MU=multi.py
run patched "" "" ""
run stock "" "" ""
run block-no-fsync $K "                    f.flush()\n                    os.fsync(f.fileno())\n" "                    f.flush()\n"
run blocks-dir-no-fsync $K "                _fsync_dir(self.blocks)               # the blocks' renames" "                pass               # the blocks' renames"
run points-dir-no-fsync-before-json $K "                small_crc = _put(self.points / f\"{key}.pt\", lambda p: torch.save(small, p), self.owner)\n                _fsync_dir(self.points)\n" "                small_crc = _put(self.points / f\"{key}.pt\", lambda p: torch.save(small, p), self.owner)\n"
run points-dir-no-fsync-after-json $K "                _put(self.points / f\"{key}.json\", lambda p: p.write_text(json.dumps(meta)), self.owner)\n                _fsync_dir(self.points)\n" "                _put(self.points / f\"{key}.json\", lambda p: p.write_text(json.dumps(meta)), self.owner)\n"
run json-not-put $K "                _put(self.points / f\"{key}.json\", lambda p: p.write_text(json.dumps(meta)), self.owner)" "                (self.points / f\"{key}.json\").write_text(json.dumps(meta))"
run put-no-fsync $K "        crc = _crc_file(tmp, sync=True)" "        crc = _crc_file(tmp, sync=False)"
run put-rename-first $K "        crc = _crc_file(tmp, sync=True)\n        os.replace(tmp, path)\n" "        os.replace(tmp, path)\n        crc = _crc_file(path, sync=True)\n"
run block-rename-before-fsync $K "                with open(tmp, \"wb\") as f:\n                    f.write(memoryview(data))\n                    f.flush()\n                    os.fsync(f.fileno())\n" "                with open(tmp, \"wb\") as f:\n                    f.write(memoryview(data))\n                os.replace(tmp, self.blocks / f\"{h}.bin\")\n                tmp = self.blocks / f\"{h}.bin\"\n                with open(tmp, \"rb\") as f:\n                    os.fsync(f.fileno())\n"
run block-crc-not-stored $K "            self.have_blocks[h] = crc\n" "            self.have_blocks[h] = 0\n"
run init-no-dir-fsync $K "        for d in (self.dir, self.dir.parent):          # the folders themselves to the disk\n            _fsync_dir(d)\n" ""
run owner-never $K "        self.owner = (st.st_uid, st.st_gid) if os.geteuid() == 0 and st.st_uid != 0 else None" "        self.owner = None"
run block-no-chown $K "            _own(tmp, self.owner)\n            os.replace(tmp, self.blocks" "            os.replace(tmp, self.blocks"
run put-no-chown $K "        write(tmp)\n        _own(tmp, owner)\n" "        write(tmp)\n"
run put-keeps-tmp $K "    except BaseException:\n        tmp.unlink(missing_ok=True)\n        raise\n    return crc" "    except BaseException:\n        raise\n    return crc"
run block-keeps-tmp $K "        except BaseException:\n            tmp.unlink(missing_ok=True)\n            raise\n        with self.lock:" "        except BaseException:\n            raise\n        with self.lock:"
run dirs-no-chown $K "            _own(p, self.owner)\n" "            pass\n"
run files-not-private $K "    os.chmod(path, 0o700 if os.path.isdir(path) else 0o600)\n" ""
run block-crc-unchecked $K "    if zlib.crc32(data) != crc:\n        raise Corrupt" "    if False:\n        raise Corrupt"
run small-crc-unchecked $K "    if _crc_file(path) != crc:\n        raise Corrupt" "    if False:\n        raise Corrupt"
run read-failure-raises $K "                self._fail(f\"block {h}: {exc}\", h)\n                continue\n            raw" "                raise\n            raw"
run size-unchecked $K "            if at != raw.numel():                     # checked before any row is written" "            if False:                     # checked before any row is written"
run failed-not-skipped $K "            if self.failed is not None:\n                self.next = upto\n                break\n" ""
run bad-block-not-named $K "                self._fail(f\"block {h}: {exc}\", h)" "                self._fail(f\"block {h}: {exc}\")"
run small-failure-raises $K "                self._fail(f\"small state: {exc}\")" "                raise"
run lineage-unchecked $K "        if [h for _, _, h in self.hashes] != m[\"blocks\"]:" "        if False:"
run crc-count-dropped $K "                if isinstance(exc, Corrupt):\n                    with self.tier.lock:\n                        self.tier.stats[\"crc_failed\"] += 1\n                self._fail(f\"block" "                self._fail(f\"block"
run load-failed-uncounted $K "                self.tier.stats[\"load_failed\"] += 1" "                pass"
run forget-keeps-points-naming-bad $K "            drop = {key} | {k for k, m in self.index.items() if any(b in bad for b in m[\"blocks\"])}" "            drop = {key}"
run forget-drops-shared-blocks $K "            gone = set(bad) | {b for b in cands if b not in named}" "            gone = set(bad) | cands"
run forget-ignores-waiting $K "            named |= {b for job in self.pending for b in job.names}\n" ""
run forget-keeps-bad-block $K "            gone = set(bad) | {b for b in cands if b not in named}" "            gone = {b for b in cands if b not in named}"
run point-with-gone-block $K "                if None in crcs:" "                if False:"
run restart-keeps-orphan-blocks $K "        for b in present - self.have_blocks.keys():" "        for b in ():"
run restart-keeps-orphan-points $K "            if f.name.split(\".\")[0] not in self.index:" "            if False:"
run restart-ids-unchecked $K "                        or zlib.crc32(raw) != m[\"ids_crc\"] or" "                        or"
run restart-crc-count-unchecked $K "                if (len(crcs) != len(blocks) or not all" "                if (not all"
run keep-builds-default-0 $K "\"TF_GLM_KV_TIER_KEEP_BUILDS\", \"\") or 1)" "\"TF_GLM_KV_TIER_KEEP_BUILDS\", \"\") or 0)"
run prune-own-dir $K "if d.is_dir() and d != self.dir)" "if d.is_dir())"
run prune-other-ranks $K "self.dir.parent.glob(f\"glm-*-r{self.rank}\")" "self.dir.parent.glob(\"glm-*\")"
run prune-none $K "        for d in others:\n            shutil.rmtree(d, ignore_errors=True)\n" "        for d in []:\n            shutil.rmtree(d, ignore_errors=True)\n"
run key-no-build $K "\"build\": build_id(g.model_dir, drafter_dir)," "\"build\": \"\","
run key-no-weights $K "\"weights\": weights_id(g.model_dir, drafter_dir)," "\"weights\": \"\","
run key-no-agreed $K "\"agreed\": repr(getattr(g, \"agreed\", None))}" "\"agreed\": None}"
run key-no-format $K "\"format\": FORMAT, \"build\"" "\"build\""
run ok-always $MU "        ok = all(r[0] for r in rows)" "        ok = True"
run follower-ignores-verdict $MU "                self._load_end(p[0], bool(p[1]))" "                self._load_end(p[0], True)"
run follower-check-no-gather $MU "                self.g._gather_ints([int(self._load_check(p[0]))])" "                self._load_check(p[0])"
run from-disk-drops-hit $MU "        got = self._load_step()\n        return got if got is not None else hit" "        return self._load_step()"
run failed-load-no-forget $MU "            self.tier.forget(ld.key, ld.bad)\n" ""
run failed-load-no-settle $MU "            self._settle(x)                           # nothing kept there: removed\n" ""
run rank0-failed-still-slices $MU "        busy = self.load_ms and self._busy() and ld.failed is None" "        busy = self.load_ms and self._busy()"
run end-without-check $MU "        rows = self.g._gather_ints([int(self._load_check(kid))])\n        ok = all(r[0] for r in rows)" "        rows = [[1]]\n        ok = self._load_check(kid) or True"
run new-blocks-ignored $K "        new = [(s, e, h) for s, e, h in hashes if h not in self.have_blocks and h not in self.inflight]" "        new = list(hashes)"
run lineage-dropped $K "        hashes = block_hashes(ids, lineage)\n        new =" "        hashes = block_hashes(ids)\n        new ="
run pump-unbounded $K "    def pump(self, arena, limit: int | None = PUMP_BLOCKS) -> None:" "    def pump(self, arena, limit: int | None = None) -> None:"
run release-noop $K "        for job in self.pending:\n            if job.where is not extent" "        for job in []:\n            if job.where is not extent"
run floor-dropped $K "            if self.min_free and free - self.backlog - size < self.min_free:" "            if False:"
run backlog-cap-dropped $K "            if self.backlog + size > self.backlog_max:" "            if False:"
run floor-on-by-default $K "\"TF_GLM_KV_TIER_MIN_FREE_GIB\", \"\") or 0)" "\"TF_GLM_KV_TIER_MIN_FREE_GIB\", \"\") or 50)"
run one-tier-check-dropped $K "    if getattr(decoder.g, \"spill_cfg\", None) is not None:" "    if False:"
run best-ignores-grid $K "            if n > len(p) or (grid and n % grid):" "            if n > len(p):"
run best-ignores-head $K "            if n == len(p) and not head_ok(k):" "            if False:"
run prune-never $K "        if self._bytes() <= self.cap:\n            return" "        if True:\n            return"
run has-ignores-blocks $K "            return m is not None and all(b in self.have_blocks for b in m[\"blocks\"])" "            return m is not None"
run query-always $MU "        return all(r[0] for r in self.g._gather_ints([int(self.tier.has(key))]))" "        return self.g._gather_ints([int(self.tier.has(key))]) is not None"
run part-after-copy $MU "            self._emit(KV_PART, [kid, ld.next + k])\n            self._flush()\n            t = time.perf_counter()\n            self._load_part(kid, ld.next + k)\n" "            t = time.perf_counter()\n            upto = ld.next + k\n            self._load_part(kid, upto)\n            self._emit(KV_PART, [kid, upto])\n            self._flush()\n"
# Not mutated (no deterministic check tells them apart): the writer's f.flush() before its fsync (a single write of a
# whole block goes past Python's buffer, so the file is complete either way), and the CRC computed on the hasher thread
# rather than the writer's (a speed choice; both are off the engine loop, which the check does verify).
rm -f "$OUT"
echo "exit $bad"
exit $bad
