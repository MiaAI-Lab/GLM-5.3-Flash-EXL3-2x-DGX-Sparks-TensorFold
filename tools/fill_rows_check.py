#!/usr/bin/env python3
"""tools/fill_rows_check.py <recipe checkout>: the FILL_ROWS setting, CPU only, in the image (the engine's code is
read and called, nothing runs on a GPU):

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD:/recipe:ro" --entrypoint python3 \
      tensorfold-glm53:v0.6.0 /recipe/tools/fill_rows_check.py /recipe

1. Every rank gets it: scripts/config.sh exports TF_GLM_FILL_ROWS=2048 by default, takes FILL_ROWS or
   TF_GLM_FILL_ROWS from the environment, and lowers the default with a smaller TF_GLM_PREFILL_ROWS; start.sh's own
   env_args puts it in ENV_ARGS, which every rank's docker run gets.
2. The engine takes it: multi.fill_rows (what MultiDecoder reads) gives 2,048 against the engine's 2,048-row prompt
   chunk at every prompt grid, and refuses a fill chunk past the prompt chunk (the engine's own refusal at startup).
3. The same memory: the prompt chunk's buffers are sized by the prompt chunk's rows, and the fill rows size nothing:
   no read of fill_rows in the GLM CUDA family sits inside an allocation or a buffer's construction, and a chunk run
   refuses more rows than the prompt chunk.
Prints PASS / FAIL lines and ALL PASS; exit 1 on any FAIL."""
import ast
import os
import pathlib
import re
import subprocess
import sys

import tensorfold.families.glm5_next.cuda as glm
from tensorfold.families.glm5_next.cuda import decode, engine, multi

recipe = pathlib.Path(sys.argv[1]).resolve()
fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    fails += not ok


# 1. every rank gets it
start = (recipe / "start.sh").read_text()
m = re.search(r"^env_args\(\) \{\n.*?^\}\n", start, re.S | re.M)
check("start.sh defines env_args", m is not None)
check("every rank's docker run takes ENV_ARGS (the workers' and rank 0's)", start.count('"${ENV_ARGS[@]}"') >= 2)


def forwarded(env, tp="3"):
    """TF_GLM_FILL_ROWS as start.sh's env_args passes it to the ranks, after scripts/config.sh, in a clean env."""
    script = (f"cd {recipe} && source scripts/config.sh >/dev/null 2>&1; TP={tp}\n"
              + (m.group(0) if m else "env_args() { ENV_ARGS=(); }\n")
              + 'env_args\nfor a in "${ENV_ARGS[@]}"; do [[ "$a" == TF_GLM_FILL_ROWS=* ]] && echo "${a#*=}"; done\n')
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/nonexistent"}
    out = subprocess.run(["bash", "-c", script], env={**base, **env}, capture_output=True, text=True, timeout=60)
    return out.stdout.split()


for env, want, what in (({}, ["2048"], "by default"), ({"FILL_ROWS": "1024"}, ["1024"], "FILL_ROWS=1024"),
                        ({"TF_GLM_FILL_ROWS": "512"}, ["512"], "TF_GLM_FILL_ROWS=512"),
                        ({"TF_GLM_PREFILL_ROWS": "1024"}, ["1024"], "TF_GLM_PREFILL_ROWS=1024 (the default follows)"),
                        ({"TF_GLM_PREFILL_ROWS": "4096"}, ["2048"], "TF_GLM_PREFILL_ROWS=4096")):
    got = forwarded(env)
    check(f"{what}: every rank gets TF_GLM_FILL_ROWS={want} (got {got})", got == want)
check("two Sparks too (TP=2)", forwarded({}, tp="2") == ["2048"])

# 2. the engine takes it
check(f"the engine's prompt chunk is 2,048 rows (decode.PREFILL_ROWS {decode.PREFILL_ROWS})", decode.PREFILL_ROWS == 2048)
check("TensorFold's own default is 1,024 (what the recipe changes)", multi.fill_rows(2048, 0, "") == 1024)
grids = (0,) + tuple(engine.GRIDS)
check(f"2,048 is taken at every prompt grid {grids}", all(multi.fill_rows(2048, g, "2048") == 2048 for g in grids))
try:
    multi.fill_rows(1024, 0, "2048")
    refused = False
except ValueError:
    refused = True
check("a fill chunk past the prompt chunk is refused at startup (TF_GLM_FILL_ROWS=2048, 1,024-row chunks)", refused)

# 3. the same memory
files = sorted(pathlib.Path(glm.__file__).parent.glob("*.py"))
ALLOC = {"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "Buffers", "reserve", "alloc", "allocate",
         "buffer_rows", "index_ring", "mla_geometry", "CUDAGraph"}
reads, sized = 0, []
for f in files:
    tree = ast.parse(f.read_text())
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and n.attr == "fill_rows" and isinstance(n.ctx, ast.Load)):
            continue
        reads += 1
        p = n
        while p in parents:
            p = parents[p]
            if isinstance(p, ast.Call):
                fn = p.func.attr if isinstance(p.func, ast.Attribute) else getattr(p.func, "id", "")
                if fn in ALLOC:
                    sized.append(f"{f.name}:{n.lineno} in {fn}()")
check(f"the fill rows size no buffer ({reads} reads in {len(files)} files; inside an allocation: {sized})",
      reads > 0 and not sized)
src = (pathlib.Path(decode.__file__)).read_text()
check("the prompt chunk's buffers are sized by the prompt chunk's rows (Buffers(w, prefill_rows, ..., prefill=True))",
      "self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True" in src)
msrc = pathlib.Path(multi.__file__).read_text()
check("a chunk run refuses more rows than the prompt chunk (multi.py: stop - start > e.prefill_rows)",
      msrc.count("stop - start > e.prefill_rows") >= 2)
print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
