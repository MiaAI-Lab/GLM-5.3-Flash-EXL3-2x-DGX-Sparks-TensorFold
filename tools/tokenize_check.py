#!/usr/bin/env python3
"""tools/tokenize_check.py <tokenizer.json> [text dir...]: patch 0099, run in the image (CPU only, the server
stopped or not).

1. Every call site: the installed server.py (parsed, not run) calls the tokenizer's `encode` nowhere and the helper
   `App._encode_ids` at every site that used to (the chat and prompt routes, /tokenize, the two forced think-close
   encodes, the tool-call gate's encode: 6 calls).
2. The same ids: the server's own `App._encode_ids` (called on a stand-in holding the real GLM tokenizer) must give
   exactly `tok.encode(text, add_special_tokens=flag).ids` for both flags over every .py file of the installed
   tensorfold package and every .md / .py / .sh file under the given dirs (the recipe checkout, say), all of them as
   one string, a long numbered filler, every added/special token's text interleaved with words, random Unicode (CJK,
   emoji, combining marks, controls) and edge cases; and, on a small template tokenizer whose special tokens the flag
   adds, that the flag passes through.
3. The GIL released: a counting thread keeps at least half its idle rate during a ~780k-token encode (`encode` holds
   the GIL and leaves it near 1%).
Prints PASS / FAIL lines and ALL PASS; exit 1 on any FAIL.

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/tokenize_check.py:/c.py:ro" \
      -v "$HF/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/<rev>/tokenizer.json:/tok.json:ro" \
      -v "$PWD:/recipe:ro" tensorfold-glm53:v0.6.0 /c.py /tok.json /recipe
"""
import ast
import pathlib
import random
import sys
import threading
import time

import tensorfold
import tensorfold.cuda.server as server
from tokenizers import Tokenizer, models, pre_tokenizers, processors

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    fails += not ok


# 1. every call site
tree = ast.parse(pathlib.Path(server.__file__).read_text())
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
held = [n.lineno for n in calls if n.func.attr == "encode" and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "tok"]
helper = [n.lineno for n in calls if n.func.attr == "_encode_ids"]
check(f"server.py calls the tokenizer's encode nowhere (lines {held})", not held)
check(f"server.py tokenizes through App._encode_ids at 6 sites (found {len(helper)})", len(helper) == 6)
encode_ids = getattr(server.App, "_encode_ids", None)
check("App._encode_ids exists", encode_ids is not None)
if encode_ids is None:
    print(f"{fails} FAIL", flush=True)
    sys.exit(1)

# 2. the same ids
tok = Tokenizer.from_file(sys.argv[1])
stand_in = type("S", (), {"tok": tok})()


def same(text):
    return all(encode_ids(stand_in, text, f) == tok.encode(text, add_special_tokens=f).ids for f in (False, True))


files = sorted(pathlib.Path(tensorfold.__file__).parent.rglob("*.py"))
for d in sys.argv[2:]:
    files += sorted(p for p in pathlib.Path(d).rglob("*") if p.suffix in (".md", ".py", ".sh") and p.is_file())
bad = [str(f) for f in files if not same(f.read_text(errors="replace"))]
check(f"{len(files)} real code and prose files, both flags", not bad and len(files) > 50)
corpus = "".join(f.read_text(errors="replace") for f in files)
check(f"all of them as one string ({len(corpus) / 1e6:.1f} MB), both flags", same(corpus))
words = ("lattice probe ferrule datum spindle gauge trellis offset stylus kinematic fixture wafer cosine baseline "
         "granite bearing encoder servo axis spline tolerance flatness runout").split()
rnd = random.Random(591)
filler = "\n".join(f"{i:06d} " + " ".join(rnd.choice(words) for _ in range(11)) for i in range(20000))
check("a numbered filler, both flags", same(filler))
special = [t.content for t in tok.get_added_tokens_decoder().values()]
mixed = [" ".join(rnd.choice(special + words + ["\n", "  ", "\t"]) for _ in range(400)) for _ in range(50)]
check(f"{len(special)} added/special tokens' texts interleaved with words (50 strings), both flags",
      all(same(t) for t in mixed) and len(special) > 0)
ranges = [(0x4E00, 0x9FFF), (0x1F300, 0x1FAFF), (0x0300, 0x036F), (0x0000, 0x001F), (0x0400, 0x04FF), (0x0600, 0x06FF)]
uni = ["".join(chr(rnd.randint(*rnd.choice(ranges))) for _ in range(rnd.randint(1, 3000))) for _ in range(200)]
check("200 random Unicode strings (CJK, emoji, combining marks, controls, Cyrillic, Arabic), both flags",
      all(same(t) for t in uni))
edge = ["", " ", "\n", "\n\n", "\t" * 50, " " * 1000, "a", "\u00a0\u2003", "<|user|>", "</think>\n"]
check("edge cases (empty, whitespace runs, single tokens), both flags", all(same(t) for t in edge))
# GLM's post-processor adds nothing (ByteLevel only), so add_special_tokens cannot change its ids: a tokenizer whose
# template does add tokens makes the flag's pass-through observable
t2 = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
t2.pre_tokenizer = pre_tokenizers.Whitespace()
t2.post_processor = processors.TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
s2 = type("S2", (), {"tok": t2})()
differs = t2.encode("hello world", add_special_tokens=True).ids != t2.encode("hello world", add_special_tokens=False).ids
check("add_special_tokens passes through (a template tokenizer: the flag changes its ids)",
      differs and all(encode_ids(s2, "hello world", f) == t2.encode("hello world", add_special_tokens=f).ids
                      for f in (False, True)))

# 3. the GIL released
big = filler * 2                                     # ~780k tokens
count, stop = [0], [False]


def spin():
    while not stop[0]:
        count[0] += 1


th = threading.Thread(target=spin, daemon=True)
th.start()
time.sleep(0.3)
c0, t0 = count[0], time.perf_counter()
time.sleep(0.5)
idle = (count[0] - c0) / (time.perf_counter() - t0)
c0, t0 = count[0], time.perf_counter()
n = len(encode_ids(stand_in, big, False))
s = time.perf_counter() - t0
rate = (count[0] - c0) / s / idle
stop[0] = True
check(f"tokenizes with the GIL released ({n} tokens in {1000 * s:.0f} ms, the other thread at {rate:.0%} of idle)",
      rate >= 0.5)
print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
