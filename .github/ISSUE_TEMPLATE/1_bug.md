---
name: Bug report
about: The model kit is not behaving as expected, is producing an error, or the numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's "Performance" and
     "Configuration" sections list the settings that affect behavior and the defaults
     shipped in scripts/config.sh. -->

- Hardware: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory each), one QSFP cable between the CX7 ports -->
- Memory available before launch, on both Sparks: <!-- `free -g`; other GPU workloads running? -->
- Image: <!-- `docker images | grep tensorfold-glm53`, or the ghcr.io tag you pulled -->
- TensorFold version: <!-- `docker exec glm53-flash-tf tensorfold --version` -->
- `start.sh` invocation: <!-- e.g. `./start.sh`, or `PARALLEL=2 ./start.sh restart --max-tokens 16384` -->
- Changed settings: <!-- environment, scripts/local.sh or .env: PARALLEL, CONTEXT, KV, DENSE, DRAFTER, DRAFT_POLICY, COPY, SPLIT, KDA_CHUNKED, VISION, COMM, TENSORFOLD_* / TF_GLM_* -->
- Startup lines: <!-- `docker logs glm53-flash-tf 2>&1 | grep -E "startup estimate|loading GLM|serving"` -->

---

## Steps to Reproduce

<!-- Please include full steps so that we can reproduce the problem. -->

1. Run `./start.sh` <!-- describe any overrides and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate the bug -->
3. ... <!-- for example "a tool call with an array parameter returns a string" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context here: a minimal request that reproduces bad output, JSON
responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.

      NOTE: the model thinks before it answers, in "reasoning_content" rather than
      "content". Budget enough max_tokens (~2,000 for anything non-trivial), or send
      "reasoning_effort": "low": with a small budget the reply is often still in its
      reasoning and "content" comes back empty on a perfectly healthy server. That is
      not a bug.

      To tell a patch bug from model behavior, send the same request with
      "draft": false (TensorFold's serial reference) and a fixed "seed": drafted and
      serial replies must be identical.
-->

```bash
curl -s http://<head-address>:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 2000
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the backticks, and mention whether it came
      from `start.sh`, `scripts/prepare.sh`, `docker logs glm53-flash-tf` (rank 0, on the
      head), `ssh <worker> docker logs glm53-flash-tf` (rank 1), or a client.

      Common culprits worth checking before filing:
        * "this start's memory budget holds a N-token window" -> another GPU workload is
          using memory on one of the Sparks (start.sh retries once with the window that fits).
        * start.sh warns "only N GiB memory available here (the default needs ~110)" -> stop other GPU
          containers on that Spark first (`docker ps`).
        * "no RoCE device or RoCE v2 GID for the link" -> WORKER is reached over another
          network: set FABRIC_PEER to the worker's CX7 address.
        * `start.sh` refuses port 8888 -> something else listens there; set PORT.
        * prepare.sh fails applying a patch -> TF_VERSION was changed; the patches are
          made for v0.6.0.
        * First start takes long -> the CUDA kernels compile once on each Spark (cached in
          ~/.cache/tensorfold-glm53); that is expected, not a hang.
-->

```

```

</details>
