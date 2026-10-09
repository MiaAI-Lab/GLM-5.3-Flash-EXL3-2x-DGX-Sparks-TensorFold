#!/usr/bin/env python3
"""tools/picture_budget_check.py <recipe checkout>: the picture budget, CPU only, in the image (TensorFold's limits
are read and called, nothing runs on a GPU):

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD:/recipe:ro" --entrypoint python3 \
      tensorfold-glm53:v0.6.0 /recipe/tools/picture_budget_check.py /recipe

1. Every rank gets it: scripts/config.sh exports TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS = MAX_IMAGES x IMAGE_TOKENS (at
   most 262,144, TensorFold's ceiling), following TENSORFOLD_GLM_MAX_IMAGES and TENSORFOLD_GLM_IMAGE_TOKENS; a value
   from the environment wins; start.sh's own env_args puts it in ENV_ARGS, which every rank's docker run gets.
2. A picture's cap does not move: under what the ranks get, GlmVisionLimits.from_env().picture_tokens(n) is the same
   for every n from 1 to max_images, so a chat's earlier pictures keep their canvases (their rows and content keys,
   which prompt reuse matches) when another picture arrives. Under TensorFold's own 16,384 it drops at the 9th.
3. The workspace covers the rows: TENSORFOLD_VISION_WORKSPACE_MIB is TensorFold's own reserve plus 8 KiB for every
   row the budget adds past 16,384, the engine's vision_workspace() reads it, and a value from the environment wins.
Prints PASS / FAIL lines and ALL PASS; exit 1 on any FAIL."""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

recipe = pathlib.Path(sys.argv[1]).resolve()
fails = 0
# the repository's defaults, not this setup's: scripts/config.sh is read from a copy without scripts/local.sh and .env
defaults = pathlib.Path(tempfile.mkdtemp())
(defaults / "scripts").mkdir()
shutil.copy(recipe / "scripts" / "config.sh", defaults / "scripts" / "config.sh")
BUDGET, WORKSPACE = "TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS", "TENSORFOLD_VISION_WORKSPACE_MIB"


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    fails += not ok


start = (recipe / "start.sh").read_text()
m = re.search(r"^env_args\(\) \{\n.*?^\}\n", start, re.S | re.M)
check("start.sh defines env_args", m is not None)
check("every rank's docker run takes ENV_ARGS (the workers' and rank 0's)", start.count('"${ENV_ARGS[@]}"') >= 2)


def forwarded(env, tp="2"):
    """The TENSORFOLD_* variables start.sh's env_args passes to the ranks, after scripts/config.sh, in a clean env."""
    script = (f"cd {defaults} && source scripts/config.sh >/dev/null 2>&1; TP={tp}\n"
              + (m.group(0) if m else "env_args() { ENV_ARGS=(); }\n")
              + 'env_args\nfor a in "${ENV_ARGS[@]}"; do [[ "$a" == TENSORFOLD_* ]] && echo "$a"; done\n')
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/nonexistent"}
    out = subprocess.run(["bash", "-c", script], env={**base, **env}, capture_output=True, text=True, timeout=60)
    return dict(line.split("=", 1) for line in out.stdout.split())


def caps(env):
    """picture_tokens(n) for n = 1..max_images as a rank with ``env`` computes it, and the engine's workspace."""
    code = ("from tensorfold.vision.glm import GlmVisionLimits as L\n"
            "from tensorfold.families.glm5_next.cuda.engine import vision_workspace\n"
            "l = L.from_env()\n"
            "print(l.max_images, vision_workspace(), *[l.picture_tokens(n) for n in range(1, l.max_images + 1)])\n")
    base = {k: v for k, v in os.environ.items() if not k.startswith(("TENSORFOLD_", "TF_GLM_"))}
    out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, timeout=300)
    if out.returncode:
        return 0, 0, []
    n, workspace, *each = (int(x) for x in out.stdout.split())
    return n, workspace, each


# TensorFold alone: the cap this recipe's default removes
n, workspace, each = caps({})
check(f"TensorFold's own budget: {n} pictures a request, 8 keep the cap and the 9th lowers it ({each[7:9]})",
      n == 50 and each[:8] == [2048] * 8 and each[8] < 2048)
own_workspace = workspace

for env, budget, what in (({}, 102400, "by default"),
                          ({"TENSORFOLD_GLM_MAX_IMAGES": "100"}, 204800, "TENSORFOLD_GLM_MAX_IMAGES=100"),
                          ({"TENSORFOLD_GLM_IMAGE_TOKENS": "1024"}, 51200, "TENSORFOLD_GLM_IMAGE_TOKENS=1024"),
                          ({"TENSORFOLD_GLM_IMAGE_TOKENS": "4096"}, 204800, "TENSORFOLD_GLM_IMAGE_TOKENS=4096"),
                          ({"TENSORFOLD_GLM_MAX_IMAGES": "4"}, 16384, "TENSORFOLD_GLM_MAX_IMAGES=4 (never below 16,384)")):
    got = forwarded(env)
    check(f"{what}: every rank gets {BUDGET}={budget} (got {got.get(BUDGET)})", got.get(BUDGET) == str(budget))
    n, workspace, each = caps(got)
    check(f"{what}: one cap for 1 to {n} pictures ({sorted(set(each))})", n > 0 and len(set(each)) == 1)
    rows = max(0, budget - 16384)
    want = own_workspace * max(4096, int(env.get("TENSORFOLD_GLM_IMAGE_TOKENS", 2048))) // 4096 + rows * 8192
    check(f"{what}: the workspace is TensorFold's plus 8 KiB a row past 16,384 ({workspace >> 20} MiB)",
          0 <= workspace - want < 2 ** 20)
check("three Sparks too (TP=3)", forwarded({}, tp="3").get(BUDGET) == "102400")

got = forwarded({"TENSORFOLD_GLM_MAX_IMAGES": "256"})
n, _, each = caps(got)
check(f"TENSORFOLD_GLM_MAX_IMAGES=256: the budget stops at TensorFold's ceiling, 262,144 (got {got.get(BUDGET)}), "
      f"which holds 128 pictures at the cap", got.get(BUDGET) == "262144" and each[127] == 2048 and each[128] < 2048)

got = forwarded({BUDGET: "16384"})
check(f"{BUDGET}=16384 from the environment wins, and the workspace is left to TensorFold",
      got.get(BUDGET) == "16384" and WORKSPACE not in got)
got = forwarded({WORKSPACE: "768"})
check(f"{WORKSPACE}=768 from the environment wins", got.get(WORKSPACE) == "768" and got.get(BUDGET) == "102400")
got = forwarded({"TENSORFOLD_GLM_IMAGE_TOKENS": "many"})
check("a value that is not a number is left for TensorFold to refuse", BUDGET not in got and WORKSPACE not in got)

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
