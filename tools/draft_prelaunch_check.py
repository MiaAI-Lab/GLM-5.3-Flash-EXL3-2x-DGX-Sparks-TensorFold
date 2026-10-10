#!/usr/bin/env python3
"""tools/draft_prelaunch_check.py: patch 0100 (a round's end launches the next round's DFlash2 block pass), in the
image with the patch applied, one GPU, one rank (about 30 s on a DGX Spark):

    docker run --rm --gpus all --network none --ipc host -v "$PWD/tools/draft_prelaunch_check.py:/c.py:ro" \
      --entrypoint python3 tensorfold-glm53:v0.6.0 /c.py

A real dflash2.Drafter over a small random checkpoint (written to a temp dir: hidden 1024, 8 query / 2 KV heads of
128, MLP 2048, 2 layers, block 8, the production ring), shared by a dflash2_multi.MultiDrafter of 4 streams whose
contexts get random taps (300 / 517 / 1,024 / 2,500 rows: the last wraps the 2,176-row ring), CUDA graphs captured
as the engine does. Then:
  1. reuse: prelaunch(streams) then candidates(same streams, pending tokens, contexts) returns the arrays a fresh
     candidates call returns, bit for bit, and launches ONE block pass (``_launch_block`` counted), for 1, 2 and 3
     streams - with the host copy zeroed and the pass held behind ~30 ms of GPU work, so a candidates that did not
     wait for the pass ahead reads the zeros;
  2. a different next call (another pending token, another stream set, another order) launches its own pass and
     returns the fresh arrays;
  3. a context changed in between - taps committed, or a restore that puts other rows under the SAME context end
     (multi.put_draft_window) - outdates the pass: candidates launches again and returns the candidates of the new
     context (which differ from the stale ones); a change of ANOTHER slot's context (taps, a restore) does not: the
     pass is reused (one launch) and its candidates are a fresh pass's, bit for bit;
  4. MultiDecoder._prelaunch decides from state every rank holds: Stream.done set by rank 0's emit alone does not
     change it; streams at their count, after an end token, with depth 0, without DFlash2 or with copied drafts are
     left out; the copies' ``proposed`` is left as it was and ``_copy_proposal`` reuses the worked-out proposal only
     while the copies' length and room are unchanged (else it proposes again); with a real copy_drafts.CopyDrafts the
     reused proposal and ``proposed`` are what a fresh ``propose`` gives;
  5. (with patch 0092's check-and-retry) a torn first copy (ids outside the vocabulary) is copied again from the
     device and gives a fresh pass's candidates without a second block pass, for an own pass and a pass ahead;
  6. TF_GLM_DRAFT_PRELAUNCH is on by default and is one of the settings the engine checks equal on every rank.
On the image without the patch it fails (the drafter has no ``prelaunch``). Prints a FAIL line per failure (an
uncaught exception prints one too), exits 1 on any, else ALL PASS."""
import json
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb),   # a raise is a FAIL line
                                   print(f"FAIL raised {t.__name__}: {v}", flush=True))

import numpy as np
import torch
from safetensors.torch import save_file

from tensorfold.families.glm5_next.cuda import copy_drafts, dflash2, dflash2_multi, multi

fails = []


def fail(msg):
    fails.append(msg)
    print("FAIL", msg, flush=True)


def check(cond, msg):
    if not cond:
        fail(msg)


dev = torch.device("cuda")
torch.manual_seed(87)
D, H, KV, HD, I, LAYERS, V, RANK = 1024, 8, 2, 128, 2048, 2, 4096, 256


def ckpt(path):
    cfg = {"hidden_size": D, "head_dim": HD, "num_attention_heads": H, "num_key_value_heads": KV,
           "rms_norm_eps": 1e-5, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
           "intermediate_size": I, "num_hidden_layers": LAYERS, "sliding_window": 2048, "is_causal": False,
           "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": V - 1,
                             "selector_rank": RANK, "selector_top_k": 16, "target_layer_ids": [1, 2, 3, 4, 5]}}
    open(os.path.join(path, "config.json"), "w").write(json.dumps(cfg))

    def r(*shape, s=0.05):
        return (torch.randn(*shape) * s).to(torch.bfloat16)
    t = {"fc.weight": r(D, 5 * D, s=0.02), "hidden_norm.weight": 1 + r(D), "norm.weight": 1 + r(D),
         "candidate_selector.hidden_projection.weight": r(RANK, D),
         "candidate_selector.predecessor_codebook": r(V, RANK, s=0.3),
         "candidate_selector.successor_codebook": r(V, RANK, s=0.3)}
    for i in range(LAYERS):
        p = f"layers.{i}."
        t.update({p + "self_attn.q_proj.weight": r(H * HD, D), p + "self_attn.k_proj.weight": r(KV * HD, D),
                  p + "self_attn.v_proj.weight": r(KV * HD, D), p + "self_attn.o_proj.weight": r(D, H * HD),
                  p + "mlp.gate_proj.weight": r(I, D), p + "mlp.up_proj.weight": r(I, D),
                  p + "mlp.down_proj.weight": r(D, I), p + "attention_conv.base_kernel": r(2, 2, D, s=0.5),
                  p + "mlp_conv.base_kernel": r(2, 2, D, s=0.5),
                  p + "attention_conv.kernel_projection.weight": r(D // 4, D),
                  p + "mlp_conv.kernel_projection.weight": r(D // 4, D),
                  p + "input_layernorm.weight": 1 + r(D), p + "post_attention_layernorm.weight": 1 + r(D),
                  p + "self_attn.q_norm.weight": 1 + r(HD), p + "self_attn.k_norm.weight": 1 + r(HD)})
    save_file(t, os.path.join(path, "model.safetensors"))


tmp = tempfile.mkdtemp(prefix="t87-")
ckpt(tmp)
head = dflash2._quantize4((torch.randn(V, D) * 0.05).to(torch.bfloat16).to(dev))
w = SimpleNamespace(device=dev, rank=0, world=1, comm=None, embed=(torch.randn(V, D) * 0.1).to(torch.bfloat16).to(dev),
                    head=head, draft_head=None, vocab_offset=0)
d = dflash2.Drafter(tmp, w, capacity=8192, tap_rows=16, ring=True)
check(d.ring == 2176, f"ring {d.ring}, expected the production 2,176 rows")
md = dflash2_multi.MultiDrafter(d, streams=4)
md.capture()
launches = []
orig_launch = md._launch_block


def counted(reqs):
    launches.append([c.slot for c, _ in reqs])
    return orig_launch(reqs)


md._launch_block = counted
ctx = md.contexts
for c, n in zip(ctx, (300, 517, 1024, 2500)):
    for start in range(0, n, 64):
        c.add_taps((torch.randn(min(64, n - start), 5 * D) * 0.5).to(torch.bfloat16).to(dev))
check([c.context_end for c in ctx] == [300, 517, 1024, 2500], f"context ends {[c.context_end for c in ctx]}")


def same(a, b):
    return all(np.array_equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q)) and len(a) == len(b)


def fresh(reqs):
    md._pre = None
    return md.candidates(reqs)


# 1. reuse
for reqs in ([(ctx[0], 17, 7)], [(ctx[1], 99, 7), (ctx[3], 5, 4)], [(ctx[0], 3, 7), (ctx[2], 4, 2), (ctx[3], 8, 7)]):
    ref = fresh(reqs)
    launches.clear()
    # the host copy poisoned (zeros: token 0, value 0) and the pass held behind ~30 ms of GPU work: candidates must
    # wait for the pass ahead (cand_ready) before it reads, else it reads the poison (a fresh pass's identical leftovers
    # in the buffer would hide a missing wait)
    md.cand_host.zero_()
    torch.cuda._sleep(60_000_000)
    md.prelaunch([(c, p) for c, p, _ in reqs])
    got = md.candidates(reqs)
    check(same(got, ref), f"reuse {[c.slot for c, _, _ in reqs]}: candidates differ from a fresh pass's")
    check(len(launches) == 1, f"reuse {[c.slot for c, _, _ in reqs]}: {len(launches)} block passes, expected 1")
    check(md._pre is None, "candidates left a pass ahead behind")
print("1. reuse checked", flush=True)

# 2. a different next call runs its own pass
base = [(ctx[1], 99, 7), (ctx[3], 5, 4)]
for name, pre, reqs in (("pending", [(ctx[1], 98), (ctx[3], 5)], base),
                        ("streams", [(ctx[1], 99)], base),
                        ("order", [(ctx[3], 5), (ctx[1], 99)], base)):
    ref = fresh(reqs)
    launches.clear()
    md.prelaunch(pre)
    got = md.candidates(reqs)
    check(same(got, ref), f"mismatch ({name}): candidates differ from a fresh pass's")
    check(len(launches) == 2, f"mismatch ({name}): {len(launches)} block passes, expected 2")
print("2. mismatches checked", flush=True)

# 3. a changed context outdates the pass
reqs = [(ctx[2], 41, 7)]
stale = fresh(reqs)
launches.clear()
md.prelaunch([(ctx[2], 41)])
ctx[2].add_taps((torch.randn(3, 5 * D) * 0.5).to(torch.bfloat16).to(dev))
got = md.candidates([(ctx[2], 41, 7)])
check(len(launches) == 2, f"taps after the pass: {len(launches)} block passes, expected 2")
check(same(got, fresh([(ctx[2], 41, 7)])), "taps after the pass: candidates are not the new context's")
# a restore under the same context end (other rows): only gen tells
end = ctx[1].context_end
rows = multi.draft_window(ctx[1], end)
ref_old = fresh([(ctx[1], 7, 7)])
other = [(torch.randn_like(r.float()) * 0.5).to(r.dtype) for r in rows]
launches.clear()
md.prelaunch([(ctx[1], 7)])
multi.put_draft_window(ctx[1], end, other)
check(ctx[1].context_end == end, "put_draft_window moved the context end")
got = md.candidates([(ctx[1], 7, 7)])
check(len(launches) == 2, f"restore under the same end: {len(launches)} block passes, expected 2")
new = fresh([(ctx[1], 7, 7)])
check(same(got, new), "restore under the same end: candidates are not the restored context's")
check(not same(new, ref_old), "the restored rows gave the old candidates (the case cannot tell a stale pass)")
# a change of another slot's context leaves the pass ahead valid
reqs = [(ctx[1], 7, 7), (ctx[3], 5, 4)]
launches.clear()
md.prelaunch([(c, p) for c, p, _ in reqs])
ctx[0].add_taps((torch.randn(5, 5 * D) * 0.5).to(torch.bfloat16).to(dev))
end2 = ctx[2].context_end
multi.put_draft_window(ctx[2], end2, [(torch.randn_like(r.float()) * 0.5).to(r.dtype)
                                      for r in multi.draft_window(ctx[2], end2)])
got = md.candidates(reqs)
check(len(launches) == 1, f"other slots changed: {len(launches)} block passes, expected 1 (the pass ahead)")
check(same(got, fresh(reqs)), "other slots changed: the pass ahead's candidates are not a fresh pass's")
print("3. outdated passes checked", flush=True)

# 4. MultiDecoder._prelaunch / _copy_proposal decide from replicated state
EOS = (7,)


class Copies:
    def __init__(self, hit):
        self.hit, self.length, self.proposed, self.calls = hit, 100, 0, 0

    def propose(self, room):
        self.calls += 1
        self.proposed = len(self.hit)
        return list(self.hit[:max(0, room)])


def lane(slot, out, count=50, dflash=True, depth=5, copies=None, done=False, stop_eos=True):
    s = SimpleNamespace(out=list(out), count=count, stop_eos=stop_eos, done=done)
    return SimpleNamespace(s=s, slot=slot, dflash=dflash, depth=depth, copies=copies, next_copy=None)


fake = SimpleNamespace(drafts=SimpleNamespace(block=8, prelaunch=None, contexts=ctx), eos=EOS)
fake._ctx = lambda l: ctx[l.slot]
fake._ends = lambda l: EOS if l.s.stop_eos else ()
seen = []
fake.drafts.prelaunch = lambda reqs: seen.append([(c.slot, p) for c, p in reqs])


def decide(lanes):
    seen.clear()
    multi.MultiDecoder._prelaunch(fake, lanes)
    return seen[0] if seen else []


cp_hit = Copies([11, 12])
cp_miss = Copies([])
lanes = [lane(0, [1, 2, 3]), lane(1, [4, 5], done=True), lane(2, [6] * 50), lane(3, [8, 7]),
         ]
got = decide(lanes)
check(got == [(0, 3), (1, 5)], f"_prelaunch picked {got}: expected slots 0 and 1 (2 at its count, 3 after an end "
                               f"token; 1's Stream.done is rank 0's alone)")
lanes = [lane(0, [1, 2, 3], depth=0), lane(1, [4], dflash=False), lane(2, [6, 9], copies=cp_hit),
         lane(3, [8, 9], copies=cp_miss)]
before = (cp_hit.proposed, cp_miss.proposed)
got = decide(lanes)
check(got == [(3, 9)], f"_prelaunch picked {got}: expected slot 3 alone (0 depth 0, 1 no DFlash2, 2 copied)")
check((cp_hit.proposed, cp_miss.proposed) == before, "_prelaunch left the copies' proposed changed")
check(lanes[2].next_copy is not None and lanes[2].next_copy[1] == [11, 12], "no worked-out copy proposal kept")
lanes[3].s.done = True
check(decide(lanes) == [(3, 9)], "Stream.done changed _prelaunch's choice")
check(decide([lane(3, [8, 7], stop_eos=False)]) == [(3, 7)], "an end token without stop_eos left the stream out")
# _copy_proposal: reuse while length and room are unchanged, else propose again
calls = cp_hit.calls
l2 = lanes[2]
c = multi.MultiDecoder._copy_proposal(fake, l2)
check(c == [11, 12] and cp_hit.calls == calls and cp_hit.proposed == 2,
      f"_copy_proposal did not reuse the worked-out proposal ({c}, calls {cp_hit.calls - calls})")
decide(lanes)
cp_hit.length += 3                       # extend ran in between
c = multi.MultiDecoder._copy_proposal(fake, l2)
check(cp_hit.calls == calls + 2, "_copy_proposal reused a proposal for changed copies")
decide(lanes)
l2.s.count -= 1                          # another room
multi.MultiDecoder._copy_proposal(fake, l2)
check(cp_hit.calls == calls + 4, "_copy_proposal reused a proposal for another room")
# with a real CopyDrafts in lockstep with an untouched twin (the path without the patch): every round's proposal and
# ``proposed`` are the twin's, through copied rounds that keep all, keep fewer (``missed``: miss_most) and find none
import copy as _copy

from tensorfold.families.glm5_next.cuda.decode import copy_room

text = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 3
real = copy_drafts.CopyDrafts(text, 2, 6, prompt=len(text), miss_most=2)
twin = _copy.deepcopy(real)
rl = lane(0, [text[-1]], count=60, copies=real, stop_eos=False)       # the text holds the fake end token 7
seen_kinds, real_calls, real_propose = set(), [0], real.propose


def counting_propose(room):
    real_calls[0] += 1
    return real_propose(room)


real.propose = counting_propose
for step, keep in enumerate((7, 1, 3, 1, 2, 4, 1)):
    missed = twin.missed
    decide([rl])
    got_c = multi.MultiDecoder._copy_proposal(fake, rl)
    want = twin.propose(copy_room(twin, rl.s.count, rl.s.out))
    seen_kinds.add(("missed" if missed else "full") if want else "none")
    check(got_c == want and real.proposed == twin.proposed,
          f"real copies round {step}: {got_c} / proposed {real.proposed}, the path without the patch {want} / {twin.proposed}")
    new = (want + [99])[:keep] if want else [text[step % len(text)]]
    for c_ in (real, twin):
        c_.extend(new)
    rl.s.out += new
check(real_calls[0] == 7, f"the real copies proposed {real_calls[0]} times in 7 rounds: the round's end's proposal "
                          "was not reused (expected one proposal a round)")
check(seen_kinds >= {"missed", "full"}, f"the real-copies rounds did not cover a missed and a full proposal: {seen_kinds}")
print("4. decisions checked", flush=True)

# 5. patch 0092's check-and-retry with the first copy issued beside the pass: a host buffer that
# reads ids outside the vocabulary after the first copy (a torn copy, simulated) is copied again from the device and
# gives a fresh pass's candidates, with no second block pass; also for a pass launched ahead
real_merge = dflash2_multi.merge_candidates
torn = [0]


def tearing_merge(*a, **kw):
    if torn[0] == 0:
        torn[0] = 1
        md.cand_host.fill_(255)                  # every id 0x437f0000 (255.0's bits): outside the vocabulary
    return real_merge(*a, **kw)


for name, ahead in (("own pass", False), ("pass ahead", True)):
    reqs = [(ctx[1], 7, 7), (ctx[3], 5, 4)]
    ref = fresh(reqs)
    torn[0] = 0
    launches.clear()
    if ahead:
        md.prelaunch([(c, p) for c, p, _ in reqs])
    else:
        md._pre = None
    dflash2_multi.merge_candidates = tearing_merge
    try:
        got = md.candidates(reqs)
    except dflash2.BadCandidates as exc:
        got = None
        fail(f"retry ({name}): the second attempt read the torn rows again: {exc}")
    finally:
        dflash2_multi.merge_candidates = real_merge
    check(torn[0] == 1, f"retry ({name}): the first attempt never merged")
    check(got is not None and same(got, ref), f"retry ({name}): candidates after the retry are not a fresh pass's")
    check(len(launches) == 1, f"retry ({name}): {len(launches)} block passes, expected 1 (a retry only copies)")
print("5. retry checked", flush=True)

# 6. the setting: on by default, and checked equal on every rank at startup (ranks that disagree would launch
# different collectives)
import ast
import inspect

from tensorfold.families.glm5_next.cuda import engine

check(multi.PRELAUNCH, "TF_GLM_DRAFT_PRELAUNCH is off in this environment (the check reads the shipped default)")
src = ast.parse(inspect.getsource(engine))
gathered = [n for n in ast.walk(src) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mine"
            for t in n.targets) and "PRELAUNCH" in ast.unparse(n.value)]
check(len(gathered) == 1, f"the engine's rank check does not include TF_GLM_DRAFT_PRELAUNCH ({len(gathered)} sites)")
check("TF_GLM_DRAFT_PRELAUNCH" in inspect.getsource(engine), "the rank check's message does not name the setting")
print("6. setting checked", flush=True)

if fails:
    print(f"{len(fails)} FAILURES")
    sys.exit(1)
print("ALL PASS")
