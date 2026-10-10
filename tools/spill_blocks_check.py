#!/usr/bin/env python3
"""Checks for patch 0110 (the spill tier's block store, tensorfold/cuda/spill_blocks.py, and its hooks in multi.py),
CPU only, run inside the image that scripts/prepare.sh built (no GPU, no model, no server); its files go to a scratch
folder under $TMPDIR:

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/spill_blocks_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

A real BlockStore over a real Arena on the host (a latents plane, div 1, and a pooled-key plane, div 4 with 2 pad
rows), with a stand-in small state:
- the block a kept state's end cuts holds the rows of the keep, not what the stream writes after it (its pooled-key
  row still filling at the keep).
Exit code 1 when a check fails; tools/spill_blocks_mutants.sh runs it on faithful mutants of each fix.
"""
import os
import sys
import tempfile
import threading
import time

import numpy as np
import torch

from tensorfold.cuda import spill as SP
from tensorfold.cuda import spill_blocks as S
from tensorfold.families.glm5_next.cuda import multi as M
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up

ROWS = 8 * ALIGN
TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "60"))
fails: list[str] = []
ROOT = tempfile.mkdtemp(prefix="spill-blocks-check-")


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


def _hung() -> None:
    print(f"FAIL timed out after {TIMEOUT:g} s", flush=True)
    os._exit(1)


threading.Timer(TIMEOUT, _hung).start()


class Snap:
    """A stand-in for decode.Snapshot: the fields ``_finish_load`` moves to the device, and the multi-stream ones
    ``_SPILL_SKIP`` leaves out."""


CLASSES = [f"{Snap.__module__}:{Snap.__qualname__}"]


def snap(ids, shared=False) -> Snap:
    s = Snap()
    s.ids, s.shared, s.lineage = list(ids), shared, ""
    s.rec, s.conv = torch.arange(64, dtype=torch.float32) + len(ids), torch.ones(8)
    s.tail, s.head, s.drafter_rows = None, None, None
    return s


def arena(seed: int) -> Arena:
    g = torch.Generator().manual_seed(seed)
    lat = torch.randint(-2**31, 2**31 - 1, (ROWS, 4), generator=g, dtype=torch.int32)
    pk = torch.randint(-2**15, 2**15 - 1, (ROWS // 4 + 2, 8), generator=g, dtype=torch.int16)
    return Arena(ROWS, [Plane(lat, 1), Plane(pk, 4, 2)])


def store(ar: Arena, name: str, rank: int = 0) -> S.BlockStore:
    cfg = SP.SpillConfig(root=os.path.join(ROOT, name), gib=1.0, min_tokens=1, min_free_gib=0.0)
    os.makedirs(cfg.root, exist_ok=True)
    return S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=rank, device="cpu", classes=CLASSES, quiet=True)


def rows(ar: Arena, base: int, n: int) -> list[torch.Tensor]:
    """The rows tokens [0, n) of an extent at ``base`` occupy, each plane (a pooled-key row still filling included)."""

    return [p.tensor[base // p.div:base // p.div + -(-n // p.div)].clone() for p in ar.planes]


def same(a: list, b: list) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def ids_of(n: int, seed: int = 0) -> list[int]:
    return [int(v) for v in np.random.default_rng(seed).integers(1, 150000, n)]


def load_rows(st: S.BlockStore, ar: Arena, key: str, base: int, n: int):
    """A read of point ``key`` into the rows at ``base`` (the store's own reader), then those rows; Nones when the
    read fails."""

    try:
        small, pt = st.finish_load(st.load_async(key, base))
    except Exception as exc:                        # noqa: BLE001  (a failed read: the checks below say so)
        print(f"     (read of {key} failed: {exc!r})")
        return None, None, None
    return small, pt, rows(ar, base, n)


# -- review 1: the block a state's end cuts is frozen at the keep ---------------------------------------------------
ar = arena(1)
pool = Pool(ROWS)
st = store(ar, "tail")
n = 2 * ALIGN + 6                                   # a tail of 6 tokens: its last pooled-key row holds 2 of 4
x = pool.add(0, align_up(n))
ids = ids_of(n)
kept = rows(ar, x.base, n)
check("tail: a state with a tail is queued", st.persist(ids, snap(ids), extent=x, lineage="a",
                                                        skip=M.MultiDecoder._SPILL_SKIP) == "queued")
g = torch.Generator().manual_seed(9)
for p in ar.planes:                                 # the stream goes on decoding in its extent past the keep
    a = n // p.div
    p.tensor[a:a + 8] = torch.randint(-100, 100, tuple(p.tensor[a:a + 8].shape), generator=g, dtype=p.tensor.dtype)
check("tail: the stream's next pooled-key row is the one the keep ended in", not same(rows(ar, x.base, n), kept))
check("tail: drained", st.drain(TIMEOUT / 2))
y = pool.add(4 * ALIGN, align_up(n))
_, pt, got = load_rows(st, ar, S.point_key(ids), y.base, n)
check("tail: the stored rows are those of the keep (the filling row as it was)", got is not None and same(got, kept))
check("tail: the tail block is not staged from the pool later", st.stats["staged_blocks"] == 2)

print("all passed" if not fails else f"FAILED: {fails}", flush=True)
os._exit(1 if fails else 0)
