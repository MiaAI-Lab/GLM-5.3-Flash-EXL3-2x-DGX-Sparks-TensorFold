## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which setting in scripts/config.sh, script behavior or patch changes
      * whether the change affects measured numbers (prefill/decode throughput, TTFT,
        concurrent requests, the startup memory estimate on rank 0) and in which direction
      * for a patch: whether it keeps the same bits, and that drafted replies still
        equal serial ones

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n start.sh stop.sh scripts/*.sh` (syntax check)
      * `python3 -m py_compile tools/*.py`
      * `shellcheck start.sh stop.sh scripts/*.sh` if available
      * `scripts/prepare.sh` (a patch change rebuilds the image on the head and copies it
        to the worker; every patch must apply with `patch -p0` against TensorFold's
        site-packages)
      * an actual launch on two Sparks, plus
        `docker logs glm53-flash-tf 2>&1 | grep -E "startup estimate|serving"`
      * `tools/needle.py` and `tools/toolcheck.py` against the running server, and
        sparkDash (https://github.com/MiaAI-Lab/sparkDash) for speed
      * if behavior changed, the measured numbers with the new settings, stating
        which configuration they came from (see README "Performance")

    If you changed a patch, confirm drafted replies still equal TensorFold's serial
    reference: the same request with "draft": false and a fixed "seed" (or
    temperature 0) must give the same reply. Say whether the patch keeps the same
    bits as before or changes prompt arithmetic (and measure quality if it does).

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [ ] A patch change keeps drafted replies equal to serial ones, and I said how I checked it (and whether its bits change).
- [ ] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [ ] If my change affects memory, I checked rank 0's startup estimate still fits at the default 4 x 1,048,576 FP8 setting.
- [ ] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
