#!/usr/bin/env python3
"""tools/vision_tokenize_check.py <tokenizer.json>: patch 0099, run in the image (CPU only, no server, no weights).

1. The call site: the installed vision/glm.py (parsed, not run) calls `self.tok.encode` nowhere and
   `encode_batch_fast` in GlmVision.prepare.
2. The same prompt: GlmVision.prepare, run on a stand-in holding the real GLM tokenizer, gives exactly the ids,
   picture positions and digest it gives with a tokenizer whose batch call is `encode` (the code before the patch),
   for prompts with 1-3 pictures (other sizes), a quoted marker, text in several scripts and every added token's text,
   and for a ~800k-token prompt with a picture; and, on a small tokenizer whose post-processor adds a special token,
   that prepare adds none (add_special_tokens stays False).
3. The GIL released: a counting thread keeps ticking during prepare on the ~800k-token prompt (its longest gap under
   50 ms; with `encode` it stops for the whole tokenization, ~1 s).
Prints PASS / FAIL lines and ALL PASS; exit 1 on any FAIL.

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/vision_tokenize_check.py:/c.py:ro" \
      -v "$HF/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/<rev>/tokenizer.json:/tok.json:ro" \
      tensorfold-glm53:v0.6.0 /c.py /tok.json
"""
import ast
import inspect
import sys
import threading
import time
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import tensorfold.vision.glm as glm

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    fails += not ok


tok = Tokenizer.from_file(sys.argv[1])

# 1. the call site
tree = ast.parse(inspect.getsource(glm))
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
encodes = [n.lineno for n in calls if n.func.attr == "encode" and ast.unparse(n.func.value) == "self.tok"]
prep = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare")
batch = [n for n in ast.walk(prep) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "encode_batch_fast"]
check(f"vision/glm.py calls self.tok.encode nowhere (lines {encodes}) and encode_batch_fast in prepare ({len(batch)})",
      not encodes and len(batch) == 1)
if fails:
    print("1 FAIL", flush=True)
    sys.exit(1)


class OldTok:
    """The tokenizer as the code before the patch used it: `encode` (the GIL held), behind the batch call's name."""
    def __init__(self, t):
        self.t = t

    def encode_batch_fast(self, texts, add_special_tokens=True):
        return [self.t.encode(x, add_special_tokens=add_special_tokens) for x in texts]


def vision(t):
    v = object.__new__(glm.GlmVision)
    v.tok, v.image_token = t, 154854
    v.limits = SimpleNamespace(picture_tokens=lambda n: 4096)
    return v


def picture(h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((3, 28 * h, 28 * w), generator=g)


def same(a, b):
    return (a.token_ids == b.token_ids and a.positions == b.positions and a.content_hash == b.content_hash
            and a.item_rows == b.item_rows and a.item_keys == b.item_keys)


SPAN = glm.IMAGE_SPAN
added = "".join(f" {t.content} word" for t in tok.get_added_tokens_decoder().values())
cases = {
    "one picture": (f"<|user|>look {SPAN} what is it?<|assistant|>", [picture(16, 16, 1)]),
    "three pictures, other sizes": (f"a {SPAN} b {SPAN} c {SPAN} d", [picture(8, 20, 2), picture(30, 4, 3),
                                                                        picture(1, 1, 4)]),
    "a quoted marker beside a picture": (f"the log said <|image|> then {SPAN} done", [picture(12, 12, 5)]),
    "scripts and added tokens": (f"中文 العربية 😀 e\u0301 {SPAN}{added}", [picture(6, 9, 6)]),
}
long = "".join(f"Line {i}: the quick brown fox, def f(x): return x * {i}\n" for i in range(18000))
cases["~800k tokens with a picture"] = (long + SPAN + long, [picture(16, 16, 7)])
new, old = vision(tok), vision(OldTok(tok))
for name, (text, pics) in cases.items():
    try:
        a = new.prepare(text, pics)
        b = old.prepare(text, pics)
        check(f"{name}: {len(a.token_ids):,} ids, positions, hash, rows and keys equal the encode path's", same(a, b))
    except Exception as e:
        check(f"{name}: raised {type(e).__name__}: {str(e)[:150]}", False)

# a tokenizer whose post-processor adds special tokens: prepare must not add them (add_special_tokens=False, as before)
small = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "a": 2, "b": 3, "<|begin_of_image|>": 4, "<|image|>": 5,
                                    "<|end_of_image|>": 6}, unk_token="[UNK]"))
small.pre_tokenizer = pre_tokenizers.Whitespace()
small.add_special_tokens(["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"])
small.post_processor = processors.TemplateProcessing(single="[CLS] $A", special_tokens=[("[CLS]", 1)])
sv, so = vision(small), vision(OldTok(small))
sv.image_token = so.image_token = 5
a, b = sv.prepare(f"a {SPAN} b", [picture(2, 2, 8)]), so.prepare(f"a {SPAN} b", [picture(2, 2, 8)])
check(f"a tokenizer that adds [CLS]: no special token added, as with encode ({a.token_ids[:3]})",
      same(a, b) and 1 not in a.token_ids)

# 3. the GIL released
text, pics = cases["~800k tokens with a picture"]
gaps, stop = [], threading.Event()


def tick():
    last = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


for label, v, want in (("encode_batch_fast", new, lambda g: g < 0.05), ("encode (before)", old, lambda g: g > 0.3)):
    gaps.clear()
    stop.clear()
    th = threading.Thread(target=tick, daemon=True)
    th.start()
    time.sleep(0.05)
    gaps.clear()
    v.prepare(text, pics)
    stop.set()
    th.join()
    check(f"prepare on ~800k tokens through {label}: the longest gap of a counting thread {max(gaps) * 1e3:.0f} ms",
          want(max(gaps)))

print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
