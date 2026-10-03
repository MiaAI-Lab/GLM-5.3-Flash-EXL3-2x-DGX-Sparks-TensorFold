<h1 align="center">GLM-5.3-Flash EXL3 on DGX Sparks with TensorFold</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<p align="center">
  <img src=".github/image.png" alt="GLM-5.3 Flash EXL3 on TensorFold, Dual DGX Sparks" width="100%">
</p>

Serve **GLM-5.3-Flash** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through an
OpenAI-compatible API, with **4 concurrent requests**, the model's full **1,048,576-token context** and **image and
video input**. It runs [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 on both Sparks (one rank on each)
in NVIDIA's PyTorch container, plus 68 patches (65 of v1.4 and 3 for 3 Sparks, experimental): DFlash2 and copy drafts, 4-bit dense weights, an FP8 KV cache,
faster prompt kernels, a one-shot RoCE all-gather between the Sparks, several requests over one shared cache pool,
vision, tool calling, `/tokenize` and `/metrics`.

- Checkpoint: [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold), Mia's AI Lab's own EXL3 quantization (routed experts at 4 bits a weight,
  BF16 elsewhere, ~176 GB), calibrated for how TensorFold serves it
  ([model card](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold))
- Drafter: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), or the checkpoint's
  own MTP head (`DRAFTER`, see [Configuration](#configuration))
- API model id: `GLM-5.3-Flash-EXL3`
- Context: **1,048,576 tokens** a request; the 4 requests share an FP8 KV pool of about **2.9M tokens** (2,922,496 at the measured start)
- Tool calling, structured outputs (xgrammar), `/tokenize`, and `reasoning_effort` `low` / `high` / `max`
- One command on the first Spark: `./start.sh` sets up both Sparks and starts both ranks; `./stop.sh` stops them

## Performance

Two DGX Sparks at the default configuration (4 streams, 1,048,576-token window, FP8 KV cache, 4-bit dense weights,
DFlash2 plus copy drafts, vision on), with the GPU clocks capped at 2,200 MHz. Decode and prefill were measured with
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) through the OpenAI API, from another machine on the network.

**Decode** (aggregate across the concurrent requests, per request, and time to first token)

| Concurrent requests | Prose | Prose, per request | TTFT | Structured | Structured, per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.4 tok/s | 60.4 tok/s | 170 ms | 114.7 tok/s | 114.7 tok/s | 149 ms |
| 2 | 79.2 tok/s | 40.4 tok/s | 269 ms | 147.6 tok/s | 77.1 tok/s | 295 ms |
| 3 | 89.5 tok/s | 30.6 tok/s | 330 ms | 196.3 tok/s | 67.4 tok/s | 317 ms |
| 4 | 108.8 tok/s | 27.9 tok/s | 340 ms | 227.9 tok/s | 61.1 tok/s | 415 ms |

Replies served 4 at a time are identical to the same requests served one at a time (11 of 11 cases staggered, and 11
of 11 sent in a burst).

**Prefill**

| Prompt | Prefill | Time to first token |
| ---: | ---: | ---: |
| 8,219 tokens | 1,952.2 tok/s | 4.21 s |
| 16,407 tokens | 1,973.5 tok/s | 8.31 s |
| 32,790 tokens | 1,978.9 tok/s | 16.57 s |
| 65,563 tokens | 1,942.5 tok/s | 33.75 s |
| 131,099 tokens | 1,837.9 tok/s | 71.33 s |
| 262,170 tokens | 1,641.7 tok/s | 159.69 s |
| 981,841 tokens (needle in a haystack) | 1,015 tok/s | 967 s, needle found |

**Prompt reuse** (the server resumes from a kept prompt state instead of prefilling, with the same reply)

| Prompt | First time | Next time |
| --- | ---: | ---: |
| An identical 64k-token prompt, sent again | 34 s | under 0.07 s |
| A new conversation with the same 7.9k-token system prompt | 4.24 s | 0.13 s |

**Quality** (FP8 KV cache and 4-bit dense weights, see [Checks](#checks))

| Benchmark | Score |
| --- | ---: |
| GSM8K (250 problems, thinking off) | 98.8% |
| HumanEval (164 problems, thinking off) | 95.7% |
| HumanEval+ and MBPP+ (542 problems, thinking on) | 86.5% |

Details on the [model card](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on their GPUs: each
  needs ~110 GiB free memory when the server starts (`start.sh` warns below that; stop other GPU work).
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private
  subnet (e.g. `192.0.2.1/24` and `192.0.2.2/24`; `ping` must work), with a RoCE v2 GID (`start.sh` checks). One QSFP
  port of a Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE devices
  (`enp1s0f0np0` / `enP2p1s0f0np0`, `rocep1s0f0` / `roceP2p1s0f0`). Address both twins on each Spark, in the link's
  subnet or each in its own (NVIDIA's two-Spark playbook gives them different ones): both are then used, as is a second
  cabled port in the link's subnet. A prompt chunk's all-gather is ~1.8x faster on two rails; one rail is one x4
  (~112 Gb/s of the port's 200).
- **Key-based ssh** from the first Spark (the head, which runs `./start.sh` and the API) to the second (the worker):
  `ssh-copy-id user@<worker>` (after `ssh-keygen -t ed25519` if you have no key); check with
  `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime, your user in the `docker` group, and `rsync`, on both Sparks.
- **Disk, on each Spark:** ~205 GB: ~176 GB (164 GiB) for the checkpoint and ~2.3 GB for DFlash2 under
  `~/.cache/huggingface`, ~25 GB for the image under Docker's root. `prepare.sh` asks for what the download still
  needs (files already in the cache, from an earlier revision say, do not count; 180 GB, `MIN_FREE_GB`, when it cannot
  ask the Hub what is missing) and 35 GB under Docker's root (`IMAGE_FREE_GB`; the sum when they share a filesystem),
  and checks the worker for what the copy must send. With `WORKER_WEIGHTS=nfs` the worker needs only the image
  ([Worker weights over NFS](#worker-weights-over-nfs)).
- Optional: the `hf` CLI on the head (faster download) and a Hugging Face token (`~/.cache/huggingface/token` or
  `HF_TOKEN`).

## Quick start

<p align="center">
  <img src=".github/ascii.png" alt="start.sh banner: TensorFold ribbon and MIA AI LAB, GLM-5.3 Flash EXL3 · Dual DGX Sparks" width="100%">
</p>

On the head:

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@192.0.2.2 in it (and FABRIC_PEER, see below)
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it: if it is on another network than the link,
set `FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of this repository.

The first run sets up both Sparks (see below): the image (~25 GB) on each, the checkpoint (~176 GB) and DFlash2
downloaded on the head and copied to the worker, then the CUDA kernels compile once per image (into `~/.cache/tensorfold-glm53/<image hash>`).
Later starts take 2 to 6 minutes to load ~80 GiB of weights on each Spark. `start.sh` shows each step and the server's
log, runs a smoke test through both ranks, and prints `GLM-5.3-Flash-EXL3 is now LIVE! on port 8888` with the endpoint.

Any OpenAI client works with `base_url = "http://<head-address>:8888/v1"` and the model `GLM-5.3-Flash-EXL3`. The model
thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`.

```bash
curl -s http://<head-address>:8888/v1/models
curl -s http://<head-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'

./start.sh restart                                       # restart both ranks, e.g. after changing a setting
./stop.sh                                                # stop both ranks and free their GPU memory
docker logs -f glm53-flash-tf                            # rank 0's log (here)
ssh <worker> docker logs -f glm53-flash-tf               # rank 1's log (why it exited, if it did)
curl -s http://<head-address>:8888/health                # busy flag, streams, free pool tokens
```

**Logs of earlier runs.** `docker rm` deletes a container's log, so `stop.sh` (and `start.sh`, before it removes a
stopped container left from an earlier run, a crashed one say) first saves each rank's log, stdout and stderr with
timestamps, gzipped, and prints where: `~/.cache/tensorfold-glm53/logs/<date>-<time>-rank0.log.gz` on the head
(`LOG_DIR`) and `~/.cache/tensorfold-glm53/logs/<date>-<time>-rank1.log.gz` on the worker. The newest 10 of each
rank are kept (`LOG_KEEP`; `0` saves none). Read one with `zcat` or `zless`; attach both when you report a crash.

If `start.sh` stops at a check:

| Message | What to do |
| --- | --- |
| `only N GiB memory available` | other GPU work runs on that Spark: stop it (`docker ps`) and restart |
| `no RoCE device or RoCE v2 GID for the link` / `no route from this node to ...` | the route to the worker does not go over the CX7 port: set `FABRIC_PEER` to the worker's CX7 address and check both ends are addressed (`ip -4 addr`) |
| `the worker's .../hub is not writable` | a container left it root-owned: fix its ownership on the worker |

## Images and video

GLM's own vision tower runs on the head (rank 0: 1.05 GiB of bf16 weights and 0.75 GiB of workspace). Send images and
videos as OpenAI-style content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<head-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`).
Pictures and clips can also be parts of a tool result (`"role": "tool"`, as agents return screenshots): the model
reads them inside that result (with `VISION=0`, it is told it cannot see them, in the template's own words). A
conversation with pictures in its history resumes from its kept prompt states like a text one, as long as the
pictures are the same.

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (`TENSORFOLD_GLM_MAX_IMAGES`), 10 MB each, 64 MB in all | up to 4 (`TENSORFOLD_GLM_MAX_VIDEOS`), 64 MB each, 96 MB in all, up to an hour of footage each |
| Tokens | at most 2,048 a picture (`TENSORFOLD_GLM_IMAGE_TOKENS`; a 1080p picture takes 2,040); a request's pictures share 16,384 (`TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS`), so past 8 each gets an equal share (327 with 50) | 2 frames a second, at most 128 frames spread over the whole clip (`TENSORFOLD_GLM_VIDEO_FRAMES`), at most 16,384 tokens a clip (`TENSORFOLD_GLM_VIDEO_TOKENS`); a request's clips share 32,768 (`TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS`), so 3 or 4 clips get 10,922 or 8,192 each |

A request body can be up to 96 MiB, so data URLs carry about 70 MB of pictures and clips in all.
By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs.
`VISION=0` serves text only and leaves the tower's ~1.8 GiB on rank 0 to the cache.

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** `scripts/prepare.sh`, when the setup is not ready on both Sparks (first run, new patches, another
   model, drafter, revision, image or worker).
2. **Checks:** the arguments (TensorFold's own parser), the link, the previous server (stopped on `restart`, or when
   only one rank is up; a stopped container left from an earlier run is removed, its log saved first), the port, free memory (a warning
   below ~110 GiB on either Spark).
3. **Launch:** rank 1 on the worker over ssh, then rank 0 and the API here, with the settings from `scripts/config.sh`.
4. **Loading:** rank 0's log as it comes, and every 15 s the elapsed time and how much of the startup estimate is on
   each Spark's GPU; if a rank stops, both ranks' last log lines and why.
5. **Smoke test:** one chat completion through both ranks, then the LIVE message and the endpoint.

A running server is left alone; `./start.sh restart` stops it only after the setup and argument check pass, so a
typo leaves it running (running requests are cut off; `stop.sh` warns). Extra arguments go to `tensorfold serve` on
both ranks after the defaults, so they win (`./start.sh restart --max-tokens 16384`); `./start.sh --help` lists all.
`FOREGROUND=1 ./start.sh` stays attached to rank 0's log and exits with its exit code (for a systemd unit), without
the progress lines, the window retry or the smoke test; when either rank ends, it stops the other one.

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight on both Sparks: Docker, the GPU, `rsync`, key-based ssh, the RoCE link, disk space.
2. The image `tensorfold-glm53:v0.6.0` on the head: TensorFold v0.6.0 with every `patches/*.patch` applied, plus PyAV
   (video decoding) and xgrammar (structured outputs), on NVIDIA's `nvcr.io/nvidia/pytorch:26.07-py3`. It first
   pulls the published image `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold:v0.6.0-<image hash>`,
   by the digest pinned in `scripts/config.sh` (`IMAGE_TAG` / `IMAGE_DIGEST`) while the patches are this release's
   (the hash covers the patches and those pip packages); after you change `patches/`, it pulls that hash's tag if
   one is published, else (or with `PULL=0`) it builds.
3. The same image on the worker: pulled, else streamed from the head (`docker save | docker load`), checked identical.
4. The checkpoint, and DFlash2 with `DRAFTER=dflash2`, downloaded into `~/.cache/huggingface` on the head at their
   pinned revisions (the checkpoint checked with `tensorfold info`), then copied to the worker with `rsync` over ssh
   and checked file by file. Both resume.

```bash
scripts/prepare.sh             # set up both Sparks without starting the server
scripts/prepare.sh --rebuild   # rebuild the image from scratch
PREPARE=1 ./start.sh restart   # force prepare.sh, then restart; PREPARE=0 skips the check
```

After changing `patches/`, `scripts/publish-image.sh` pushes the new image to GitHub Container Registry (`latest` and
`v0.6.0-<image hash>`, the tag `prepare.sh` looks for).

## KV pool and memory

GLM-5.3-Flash keeps a compressed latent cache (DSA) and the sparse indexer's pooled keys for every token, plus small
recurrent states for its linear-attention (KDA) layers. With `PARALLEL` above 1, all requests draw their per-token
caches from **one shared pool**:

| | Default |
| --- | ---: |
| Requests at once (`PARALLEL`) | 4 |
| Window per request (`CONTEXT`, the model's native maximum) | 1,048,576 tokens |
| KV precision (`KV`) | FP8 (e4m3 rows with a power-of-two scale each: half of bf16's bytes) |
| **Shared pool** (what is free at start minus `MEMORY_RESERVE_GIB` 14.5, at most `KV_POOL_GIB` 12.5 GiB a Spark beyond the window) | **2,922,496 tokens** at the measured start, the 12.5 GiB cap (~2.1-2.9M depending on free memory) |
| Rank 0's startup estimate | 88.09 GiB |
| Free memory (`MemAvailable`) at idle | 7.0 GiB on rank 0, 10.5 GiB on rank 1 |
| Lowest free memory under a 1M-token prompt | 5.6 GiB on rank 0, 9.5 GiB on rank 1 |

Any one request can grow to the full window, and the four together share the pool: e.g. one 1M-token conversation
next to one more of 1M, or next to three of ~640k. A request the pool cannot place yet waits until others finish (kept prompt states give way
first); `/health` shows `pool_tokens`, `pool_free_tokens` and the streams decoding, filling and paused.

TensorFold's budget on each Spark is `MemAvailable` at start minus a host reserve (`MEMORY_RESERVE_GIB`, 14.5 GiB here;
TensorFold's own default is a tenth of RAM). The server uses about 10 GiB beyond its own estimate at its peak, so the
reserve also sets the lowest free memory.
On the Spark's unified memory, running out tends to freeze the machine rather than fail an allocation. A setting that
does not fit is refused before any weights load, with the largest window that fits; `start.sh` then restarts once
with that window and says so (free memory on both Sparks for the full one). Other settings' windows:

| Setting | Window | Note |
| --- | ---: | --- |
| `KV=bf16` | 196,608 | the exact cache; 163,840 with `VISION=1` and `DENSE=fp8` or `bf16` (the tower costs rank 0 ~66k tokens of window) |
| `KV=bf16 DRAFTER=mtp` | 524,288 | one request at a time |
| `CONTEXT=0` | the largest that fits | no memory is left to keep other conversations' prompts |

## Worker weights over NFS

By default the worker keeps its own copy of the checkpoint and DFlash2 (~166 GiB, copied over the link by
`prepare.sh`). With `WORKER_WEIGHTS=nfs` it keeps none: rank 1 reads the head's Hugging Face cache over NFS, read-only.
Loading is as fast as from the worker's own disk (both ranks were live in ~2.2 minutes).

1. On the head, export the cache to the worker once (this needs root; the worker needs nothing installed):

   ```bash
   sudo apt install nfs-kernel-server
   echo "$HOME/.cache/huggingface <worker CX7 address>(ro,no_subtree_check)" | sudo tee -a /etc/exports
   sudo exportfs -ra
   ```

2. Put `WORKER_WEIGHTS=nfs` in `scripts/local.sh` or `.env`, and run `./start.sh restart`.

`prepare.sh` then creates a read-only docker NFS volume on the worker (`NFS_VOLUME`, default `glm53-hf`, no sudo) and
checks that the worker sees every file of both snapshots as the head has them, instead of copying. `NFS_PATH` is the
exported path as the worker mounts it (default: the head's `HF_CACHE`; `/` for an NFSv4 export with `fsid=0`), and
`NFS_SERVER` the head's address (default: its address on the link).

## 3 Sparks (experimental)

`./start-tp3.sh` runs the same recipe as tensor parallel over three Sparks (`TP=3` with `./start.sh`'s options;
`./stop.sh` stops every configured worker). **The engine for `--tp N` comes from this repo's patches 0066-0068**
(TensorFold v0.6.0 itself serves two ranks only), in the same published image as two Sparks (`prepare.sh` pulls it on the head and copies it to every worker). **Three Sparks** were tested on v1.3.2's patches (exact against two Sparks, drafted == serial, images and
tool calls, concurrent requests). On top of v1.4 (2026-10-03, `KV_POOL_GIB=27`, NCCL, two boots: sliced fill on and
off): concurrent requests equal one at a time (22/22 each boot), drafted == serial (6/6), long-prompt replies and every
streamed reply the same with the sliced fill on and off, the 195k needle right (prefill 115.0 s), a prompt cancelled
mid-fill gone in 0.16 s with the other replies unchanged, tool calls whole. Prefill 12k / 50k / 149k: 6.2 / 25.7 / 84.2 s
(two Sparks: 6.5 / 26.8 / 87.1). Streaming, gaps p50 / p90 between a reply's events: one reply 15 / 18 ms, four at
once 34 / 46 ms; three replies while a ~25k-token prompt fills 81 / 134 ms with the sliced fill (134 / 400 ms with
`FILL_BUDGET_MS=0`). Rank 0 had 10.6 GiB free idle and 8.4 GiB at its lowest (a 149k prompt); at the default 32 GiB
it had 5.8 GiB free idle, below the ~10 GiB this recipe keeps under load, so these runs used 27.

**Three Sparks, measured with sparkDash** (the default configuration otherwise: 4 streams, 1,048,576-token window, FP8 KV
cache, 4-bit dense weights, DFlash2 plus copy drafts, vision on)

| Concurrent requests | Prose | Prose, per request | TTFT | Code | Code, per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 77.6 tok/s | 77.6 tok/s | 144 ms | 104.3 tok/s | 104.3 tok/s | 192 ms |
| 2 | 95.7 tok/s | 49.5 tok/s | 264 ms | 139.0 tok/s | 72.7 tok/s | 293 ms |
| 3 | 121.2 tok/s | 42.0 tok/s | 249 ms | 158.7 tok/s | 55.5 tok/s | 379 ms |
| 4 | 146.2 tok/s | 37.3 tok/s | 262 ms | 169.6 tok/s | 45.7 tok/s | 348 ms |

| Prompt | Prefill | Time to first token |
| ---: | ---: | ---: |
| 8,212 tokens | 2,000.6 tok/s | 4.59 s |
| 16,408 tokens | 2,000.4 tok/s | 8.20 s |
| 32,790 tokens | 2,064.2 tok/s | 15.89 s |
| 65,560 tokens | 2,003.6 tok/s | 32.72 s |
| 131,094 tokens | 1,881.7 tok/s | 69.67 s |
| 262,169 tokens | 1,654.5 tok/s | 158.45 s |

Against two Sparks ([Performance](#performance)): prose decode 60.4 -> 77.6 tok/s for one request and 108.8 -> 146.2
tok/s for four at once; prefill about the same (1-4% faster).

`DRY_RUN=1 ./start-tp3.sh` shows what would run (every rank's `docker run`, the links found, nothing stopped or
started). It uses NCCL for every all-gather by default (`COMM=nccl`); `COMM=roce ./start-tp3.sh` sends the small
ones over RoCE, each peer on the devices that share its subnet (the TP-N engine's RoCE; `TF_ROCE_HCA` lists a node's
devices toward all its peers, and each device's RoCE v2 GID is found on its own).

- **Cabling.** A triangle, one direct QSFP cable per pair, each cable its own subnet (both CX7 ports of
  each Spark are used, each toward one peer). Wire it as a directed ring, each Spark's port 0 to the next Spark's
  port 1; a mirrored ring leaves one pair NCCL cannot connect. `start.sh` refuses a pair of ranks without a common
  subnet.
- **Settings** (in `scripts/local.sh`): `WORKER` (rank 1) and `WORKER2` (rank 2), each with its own `FABRIC_PEER`,
  `FABRIC_PEER2` if needed (`./start.sh` uses `WORKER` only). Weights per worker: `WORKER_WEIGHTS`, `WORKER_WEIGHTS2`
  (default: `WORKER_WEIGHTS`). With `nfs`, each worker mounts the head's export from the head's address on that
  worker's own link (`NFS_SERVER`, `NFS_SERVER2` override it), so the export must allow every worker's link address
  or subnet. `prepare.sh` says which worker cannot mount it. A worker whose Hugging Face cache is not its `HF_HOME`:
  `WORKER_HF_CACHE`, `WORKER_HF_CACHE2`.
- **Every worker as the one worker at two Sparks:** `prepare.sh` checks each worker's image by content (#8), compares
  each copy's file list in byte order (#21) and counts only what `rsync` must send (#24); `start.sh` and `stop.sh`
  save every rank's log (`<date>-<time>-rank<N>.log.gz` in `~/.cache/tensorfold-glm53/logs` on that worker).
- **Links.** At two Sparks the link and its rails are found as before (both PCIe twins of a cabled port, #30). Past
  two, the nodes' RoCE devices are paired by subnet: a twin is used when it has an address in a subnet its peer
  shares; one without an address is not added by name as at two Sparks.
- **What it sets.** The rendezvous (`--master`) is the head's LAN address, what its hostname resolves to (`MASTER_ADDR`
  overrides it); every worker must route to it. NCCL's bootstrap socket uses each node's default-route netdev
  (`SOCKET_IFNAME` overrides it), since no CX7 port reaches both peers of a triangle. Data stays on RoCE:
  `NCCL_IB_HCA` lists every CX7 device toward every peer (found from the subnets the nodes share), with
  `NCCL_CROSS_NIC=1`, `NCCL_IB_SUBNET_AWARE_ROUTING=1` (the image's NCCL 2.30.7 has it), the RoCE v2 GID index when
  it is the same on all of a node's devices, and no P2P or SHM transport. These come from a vLLM recipe that ran on
  the same three Sparks.
- `prepare.sh` on its own takes `TP` too: `TP=3 scripts/prepare.sh`.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Set one for a single run from the environment
(`PARALLEL=2 ./start.sh restart`), or keep it in `scripts/local.sh` (sourced as bash) or in a `.env` file next to
`start.sh` (plain `KEY=value` lines, read, never run); both files are yours, not the repository's. The first that
sets a value wins: the environment, then `scripts/local.sh`, then `.env`, then the default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKER` / `FABRIC_PEER` | empty | the worker's ssh target (`user@<address>` or `user@<host name>`), and its CX7 address when `WORKER` is on another network |
| `WORKER_HF_CACHE` | the worker's `HF_HOME` | the worker's Hugging Face cache, when it is not its `HF_HOME` (e.g. a shared models folder) |
| `MASTER_PORT` | `29551` | the ranks' rendezvous port (keep it on the private link) |
| `TP` / `WORKER2` | `2` / empty | Sparks in all (`2`, or `3` through `./start-tp3.sh`), and the ssh target of rank 2 ([3 Sparks](#3-sparks-experimental)); `FABRIC_PEER2`, `WORKER_HF_CACHE2`, `WORKER_WEIGHTS2`, `NFS_SERVER2` as the worker's own |
| `MASTER_ADDR` / `SOCKET_IFNAME` | see [3 Sparks](#3-sparks-experimental) | the rendezvous address (`TP=2`: the head's address on the link) and, with `TP` above 2, NCCL's bootstrap netdev |
| `PARALLEL` | `4` (`1` with `DRAFTER=mtp`) | requests decoded together, 1 to 4 (above 1 needs `DRAFTER=dflash2`) |
| `CONTEXT` | `1048576` | prompt + reply window per request (with `KV=fp8`; other defaults in [KV pool and memory](#kv-pool-and-memory)); `0`: the largest that fits |
| `KV` | `fp8` | `fp8` or `bf16` (exact, shorter window) DSA latent cache and indexer keys |
| `WORKER_WEIGHTS` | `copy` | `copy`: the worker keeps its own copy of the weights; `nfs`: it reads the head's over NFS ([Worker weights over NFS](#worker-weights-over-nfs)); with `NFS_PATH`, `NFS_SERVER`, `NFS_VOLUME` |
| `KV_POOL_GIB` / `MEMORY_RESERVE_GIB` | `12.5` (`32` at `TP=3`) / `14.5` | the shared pool beyond the window (kept prompts, more long conversations at once) grows into what is free at start minus the reserve, up to `KV_POOL_GIB` GiB a Spark; the reserve sets the lowest free memory on the head (~4.5-5 GiB); raise it when other work shares the Sparks |
| `DENSE` | `q4` | the checkpoint's BF16 weights (attention, shared experts, dense layers, head): `q4` (4-bit groups of 64, the head in FP8, kv_b in BF16), `fp8` or `bf16`. **Non-English prompts:** `q4` can lose the end of turn on short French coding prompts (replies run to `max_tokens`, issue #18); `fp8` keeps it, at ~10% decode speed |
| `DRAFTER` | `dflash2` | `dflash2`: IncoAI's DFlash2 drafter, licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), **non-commercial use only**; +5-10% decode. `mtp`: the checkpoint's own MTP head, one request at a time, which avoids that license (set it before the first `./start.sh` and DFlash2 is never downloaded) |
| `TF_GLM_MTP` | `auto` | the checkpoint's MTP head beside DFlash2: `auto` leaves it out while DFlash2 drafts every request; `1` (TensorFold v0.6.0's own default) loads it, 1.77 GiB a Spark, with `PARALLEL=1`. `DRAFTER=mtp` always loads it |
| `DRAFT_POLICY` | `fnc7:0.3` | how many DFlash2 drafts a round verifies: up to 7, until the drafts' chance under the request's own sampling noise drops below 0.3 |
| `COPY` / `COPY_MAX` | `1` / `15` | copy drafts: when the reply's last 8 tokens occurred before, the tokens that followed are verified ahead of DFlash2's, up to 15 a round |
| `COPY_CODE` | `1` | 16-row verify windows as CUDA graphs, and copies from the reply itself only after a 16-token match |
| `SHARED_PREFIX` | `1` | conversations that share a system prompt reuse its prompt state |
| `MAX_TOKENS` | `32768` | the reply budget (reasoning and answer) of a request that sets no `max_tokens`; TensorFold's own default is 4,096 |
| `THINKING` | `1` | think before answering by default; `0` answers directly unless a request asks to think |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video input; `1` also accepts public `https://` URLs |
| `COMM` | `roce` | the ranks' small all-gathers as one-shot RDMA writes over the RoCE link; `nccl`: NCCL for all |
| `SPLIT` | `1` | prompt chunks' hyper-connection work split between the Sparks, its exchanges overlapped with the next rows' work |
| `KDA_CHUNKED` | `1` | KDA prompt chunks in chunked (WY) form, one CUDA kernel |
| `TF_GLM_L2PF` | `1` | L2 prefetch in decode: the weights the next kernels read brought into L2 during each layer's all-gathers; `0`: off |
| `TF_GLM_EXL3_LOADS` | `nc` | the decode expert kernel's trellis as 16-byte non-coherent loads a step ahead (`nc2` / `nc4`: 2 or 4 steps); `0`: TensorFold's 32-bit loads |
| `TF_ROCE_MAX_KB` | `512` | the largest all-gather (KiB) sent over RoCE with `COMM=roce` (TensorFold's default: 256); 512 covers the 17-32-row verify windows of concurrent requests |
| `TF_ROCE_WAIT_S` | `20` | seconds a RoCE all-gather waits for the other Spark before it fails (1 to 3600); until the engine is built, each gather first meets the peer in an NCCL barrier, so a long first start (kernels compiling) does not run into it |
| `TF_GLM_MULTI_LONE` | `0` | `1`: with `PARALLEL` above 1, a request decoding alone runs on the one-stream graphs (+0.6-0.9%), but its move to the pool's first rows evicts other conversations' kept prompts there (issues #12, #13); `0`: on the batched ones |
| `TF_GLM_CACHE_ENTRIES` | `32` | kept prompt states at most (TensorFold's default is 8); past this count a conversation's earlier (superseded) states go first, then the least recently used, however much of the pool is free (patch `0063`, @Alexbob0). An agent request keeps 1 to 3 (issue #17); each costs ~45 MiB at start |
| `TF_GLM_MULTI_WATCHDOG_S` / `TF_GLM_MULTI_WATCHDOG_EXIT` | `300` / `0` | a rank stuck this many seconds in one step prints every thread's stack to its log (`/health`'s `iteration_s` shows how long the current step has run); `EXIT=1`: the server then exits (code 1) instead of waiting, so a supervisor can restart it. Both ranks also check that they received the same message each step, and stop with an error naming it if not |
| `TF_GLM_CLEAR_THINKING` | `0` | `1`: drop earlier turns' reasoning from the prompt, as the checkpoint's template does (agents then prefill the previous turn's tool loop again); a request's `chat_template_kwargs.clear_thinking` wins |
| `MULTI_PREFILL` | `1` | prompts that arrive together are filled in one forward (each with the bits it gets alone); `0`: one after another |
| `STREAM_SMOOTH` / `STREAM_SMOOTH_MS` | `1` / `400` | streamed replies go out one token an event at a steady pace from a playout buffer of this many ms, so drafted bursts (~3 tokens every ~50-100 ms) and short pauses do not show; text appears that much later, the reply ends when it did, tool calls keep their place; `0`: one event a round |
| `FILL_BUDGET_MS` | `200` | while other requests decode, a new prompt fills in layer slices of about this many ms with a decode round between them; `0`: whole 1,024-row chunks, each of which froze the other replies ~0.65 s. A request alone fills at full speed either way |
| `FILL_DRAFTS` | `1` | the rounds between those slices draft as usual (the other replies ~10 tokens/s each during a fill); `0`: one token a reply (the new prompt's first token sooner, the other replies slower while it fills) |
| `SERVED_NAME` / `PORT` / `HOST` | `GLM-5.3-Flash-EXL3` / `8888` / `0.0.0.0` | the model id in `/v1/models` and replies; where the API listens |
| `TENSORFOLD_GLM_IMAGE_TOKENS` / `_VIDEO_TOKENS` / `_VIDEO_FRAMES` | `2048` / `16384` / `128` | a picture's and a clip's token caps, and a clip's frames |
| `TENSORFOLD_GLM_MAX_IMAGES` / `_MAX_VIDEOS` | `50` / `4` | pictures and clips a request |
| `TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS` / `_REQUEST_VIDEO_TOKENS` | `16384` / `32768` | tokens a request's pictures and clips share (each still within its own cap) |
| `PREPARE` / `PULL` | `auto` / `1` | `start.sh` runs `scripts/prepare.sh` when needed (`1` always, `0` never); `prepare.sh` tries the prebuilt image first (`0`: always build locally) |
| `WAIT_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `30` | seconds `start.sh` waits for the server, and `stop.sh` gives it to shut down |
| `LOG_DIR` / `LOG_KEEP` | `~/.cache/tensorfold-glm53/logs` / `10` | where `stop.sh` saves rank 0's log before it removes the container (rank 1's, and ranks 2 and 3's at `TP` above 2, to the same folder on their worker), and how many of each rank's to keep ([Logs of earlier runs](#quick-start)); `0`: none |

Speed settings' measured effects: [What the patches change](#what-the-patches-change). Any `TENSORFOLD_*`,
`TF_GLM_*` or `TF_ROCE_*` variable is passed to both ranks (export it in `scripts/local.sh`; `.env` lines are).

**Revision pins.** The checkpoint and DFlash2 are pinned to the revisions this recipe was measured with
(`MODEL_REVISION`, `DFLASH2_REVISION`): both ranks serve exactly those from the local cache, and a new upstream commit
changes nothing until the pin does. Set one empty to take the Hub's `main` when first downloaded.

Less common settings are described in `scripts/config.sh` and `scripts/nodes.sh`: `MODEL_ID`, `DFLASH2_ID`,
`TF_VERSION`, `TF_REPO`, `BASE_IMAGE` (the patches are made for TensorFold v0.6.0; after changing any of these run
`scripts/prepare.sh --rebuild`), `IMAGE`, `CONTAINER_NAME`, `GHCR_IMAGE`, `HF_CACHE` (default `$HF_HOME` or
`~/.cache/huggingface`), `KERNEL_CACHE`, `STATE_DIR`, `MIN_FREE_GB`, `IMAGE_FREE_GB`, `NCCL_RAILS` (`1`: one RoCE
port even when the cabled port's two PCIe links, or a second port, are up), `NCCL_CHANNELS` (4), `NCCL_DEBUG`, `RSYNC_OPTS`. `start.sh` also takes `HF_HUB_OFFLINE=0` (let the
server reach Hugging Face; by default it serves from the local cache only).

### Thinking and sampling

The checkpoint's own sampling defaults apply (temperature 1.0, top_p 0.95). Per request:

- `temperature`, `top_p`, `top_k`, `min_p` and `seed` override them (`temperature: 0` decodes greedily). Without a
  `seed`, the sampler's key comes from the prompt, so the same request gives the same reply.
- `reasoning_effort`: `"low"`, `"high"` or `"max"`, at the top level or in `chat_template_kwargs` (`"medium"` is
  `"high"`, `"xhigh"` is `"max"`). Without it, GLM's template uses `max`; `"none"` (or
  `"chat_template_kwargs": {"enable_thinking": false}`) answers without thinking. `chat_template_kwargs.thinking`
  (`true` / `false`, or `{"type": "enabled"}` / `{"type": "disabled"}`, as DeepSeek-V4 clients such as pi send it) is
  read as `enable_thinking` when that is absent; other values leave the server's default.
- Earlier turns' reasoning stays in the prompt (as in zai-org's current template), so an agent's previous tool loop
  is resumed from the cache at a new user message; `"chat_template_kwargs": {"clear_thinking": true}` or
  `TF_GLM_CLEAR_THINKING=1` drops it, as the checkpoint's template does.
- The reasoning comes back in `reasoning_content`, the answer in `content`. A request without `max_tokens` gets
  32,768 tokens for both (`MAX_TOKENS`; cut to what the window has left, never refused). An empty `content` means the model
  thought until `max_tokens`: give more, or use `reasoning_effort: "low"`.

### API notes

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/tokenize` and
  `/detokenize` (also under `/v1/`), `/health`, and Prometheus `/metrics` (TensorFold v0.6.0's request counters,
  latency and time-to-first-token histograms, plus `/health`'s figures as `tensorfold_health:` metrics). No Anthropic
  `/v1/messages`.
- **Context limits:** a request whose prompt plus `max_tokens` does not fit the window is refused with HTTP 400 in
  OpenAI's wording, `"code": "context_length_exceeded"` and `param` naming the field (`messages` or `prompt`).
- **Tool calling:** `tools` / `tool_calls`, each call streamed whole once it is written (empty deltas every 2 s
  meanwhile, for clients with an idle timeout; a reply that ends inside a call, at its token limit, ends with
  `length` and never sends that call; a call the model ends without its `</tool_call>` is closed and sent when it
  then parses, else returned as text; a missing `<arg_key>` is put back), with arguments typed by their schema (an array as a JSON array,
  `null` and `const` values as such). Calls written inside the think block count when the reply ends on them, and a
  tool-calling step's reasoning is put back into the next request when an agent client drops it
  (`TF_GLM_KEEP_REASONING=0` turns that off). A past tool call whose arguments are not a JSON object is left out of
  the prompt with its result (and logged) instead of failing the request.
- **Structured outputs:** `response_format` (`json_object` or `json_schema`) and the `guided_json` / `guided_regex` /
  `guided_choice` / `guided_grammar` / `structured_outputs` fields, enforced with xgrammar (after the think block).
- **`/tokenize` / `/detokenize`:** vLLM's fields; a `prompt` or `messages` (rendered as the chat route does) to token
  ids with their `count` and `max_model_len`, and back.
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (this engine returns no token probabilities) and `n`
  other than 1. Accepted but ignored, with no error: `logit_bias`, and the presence, frequency and repetition
  penalties. `priority: "background"` serves a request after the others.
- Prompt reuse: a new prompt that extends a recent one token for token resumes from its kept state (with pictures
  and clips, only when they are the same ones), and a shared system prompt's state is reused (`SHARED_PREFIX`).
- A POST to an unknown route is answered 404 after its body is read, so a kept-alive connection stays usable; a
  client that leaves is noticed however many connections the server holds.

## What the patches change

`scripts/prepare.sh` bakes every `patches/*.patch` into the image (diffs against TensorFold v0.6.0's site-packages,
applied with `patch -p0` in filename order); `start.sh` rebuilds or re-pulls the image when the patches change.

| Area | Patches | Change | Effect |
| --- | --- | --- | --- |
| Weights | `0002-glm-dense-fp8`, `0005-glm-dense-q4` | the checkpoint's BF16 dense weights in FP8, or 4-bit with MSE-searched ranges (`DENSE`) | q4 over fp8: prose 38.9 -> 44.4 tok/s, code 44.2 -> 48.7, prefill ~1,090 -> ~1,260 tok/s |
| KV cache | `0038-glm-kv-fp8` | the DSA latent cache and the indexer's pooled keys as FP8 rows (`KV=fp8`) | the 1M window with 4 requests fits |
| Prompt | `0001-glm-exl3-prompt-experts`, `0004-glm-prompt-kernels`, `0009-glm-prefill-kernels`, `0020-glm-prompt-experts-order`, `0024-glm-prompt-select-rows`, `0028-glm-lean-prompt-scratch` | EXL3 expert kernels that keep a prompt chunk's rows in L2, launched in a better order; each row's input rotated once; dense attention only where the sparse pass needs it; token selection in blocks of 512 rows; smaller prompt scratch | faster prefill, less memory at 1M |
| Prompt, two Sparks | `0010-glm-hc-split`, `0033-glm-prefill-overlap2`, `0017-glm-overlap-priority`, `0022-glm-overlap-normal-priority`, `0064-glm-split-connect-early` | hyper-connection glue split by rows between the Sparks, exchanges overlapped with the next rows' work (`SPLIT`); the split's send/receive connection opened before the weights and the cache pool (issue #36) | 50k prefill ~1,270 -> ~1,730 tok/s with 0009 and 0020; decode rounds pay ~1.5% |
| Prompt, KDA | `0012-glm-kda-chunked`, `0014-glm-kda-chunked-gb10`, `0039-glm-kda-chunked-kernel` | the linear-attention layers' prompt chunks in chunked (WY) form, one CUDA kernel (`KDA_CHUNKED`) | 50k prefill 29.3 -> 26.4 s, 149k 91.1 -> 84.5 s (one start each) |
| Prompt reuse | `0008-glm-prompt-grid`, `0015-glm-shared-prefix`, `0042-glm-prompt-replay`, `0054-glm-image-prompt-reuse`, `0060-glm-keep-thinking`, `0063-glm-kept-cap-superseded-first`, `0071-glm-shared-prefix-copy` | prompt chunks and kept states on a token grid; a shared system prompt's state reused (copied out of the conversation that wrote it, which keeps its own state: #43); an identical prompt resumes from its kept end state; prompts with pictures resume too, their kept states told apart by each picture's content (by abhicnv007, issue #11); earlier turns' reasoning kept in the prompt (`TF_GLM_CLEAR_THINKING`, by kky42, #23); past the kept-state cap a conversation's superseded states go first (by Alexbob0, #32) | [Performance](#performance) |
| Decode | `0013-glm-decode-rounds`, `0016-glm-decode-kernels`, `0019-glm-decode-kernels2`, `0031-glm-decode-index-regs` | verify windows of up to 16 rows; faster decode matmuls, hyper-connection mixing and indexer scoring | faster decode rounds; long copy drafts (`COPY_MAX` 15: edit replies 84 -> 119 tok/s) |
| Decode, memory | `0046-glm-l2-prefetch`, `0047-glm-exl3-decode-loads` | the next kernels' weights prefetched into L2 during each layer's all-gathers (`TF_GLM_L2PF`, adapted from jayleaton/glm53-tensorfold-spark's patch 0460); the routed experts' trellis as 16-byte non-coherent loads a step ahead (`TF_GLM_EXL3_LOADS`, adapted from its patch 0580) | with `TF_ROCE_MAX_KB=512`: one request's prose 48.36 -> 49.68 tok/s, code 59.54 -> 61.49 (+2.7% / +3.3%); 4 at once prose 74.8 -> 76.6, code 100.0 -> 102.7 (two starts each); the same bits |
| Decode, indexer | `0043-glm-visible-pools` | a decode row's token scoring and split selection bounded to the pools it can see (TensorFold v0.6.0 bounds its one-program selection, PR #140) | the same tokens |
| Link | `0006-cuda-roce-allgather`, `0052-cuda-roce-startup` | the small all-gathers as one-shot RDMA writes over RoCE (`COMM=roce`; up to 512 KiB, `TF_ROCE_MAX_KB`: code at 4 streams +1.4%); until the engine is built, each eager gather first waits for the peer in an NCCL barrier, and the RoCE kernel builds during setup (`TF_ROCE_WAIT_S`, default 20 s; errors print the proxy's counters); an idle rank 1 waiting on a socket is TensorFold v0.6.0's (#132) | 11 us a 16 KiB gather against NCCL's 45, decode +6%; an idle server holds no GPU or CPU core; a first start whose ranks drift apart while building kernels no longer fails |
| Drafts | `0007-glm-copy-drafts`, `0032-glm-code-copy-drafts`, `0018-glm-noise-policies`, `0021-glm-dflash-policy-env`, `0025-glm-dflash2-ring` | copy (prompt-lookup) drafts ahead of DFlash2's (`COPY`, `COPY_CODE`); DFlash2 stop rules aware of the sampling noise (`DRAFT_POLICY`); kept prompt states' DFlash2 window with shared prefixes (the ring itself and `TF_GLM_MTP` are TensorFold v0.6.0's; the recipe sets `TF_GLM_MTP=auto`, so the MTP head is not loaded when DFlash2 drafts) | `COPY`: quote / edit replies 80.3 -> 84.4 tok/s; `COPY_CODE`: code 55.4 -> 56.2, edit 120 -> 125; `DRAFT_POLICY` over `fc5:0.3`: prose 47.2 -> 50.6 tok/s, code 51.7 -> 55.5; memory for the window |
| Drafts, tooling | `0011-glm-draft-sim` | records of DFlash2's drafts for an offline simulator of stop rules (`TF_GLM_DRAFT_DUMP`, off) | how the stop rules were tuned |
| Concurrent requests | `0026-glm-multi-kda`, `0027-glm-multi-dflash2`, `0029-glm-multi-dsa`, `0030-glm-multi-stream-engine`, `0035-glm-multi-rounds`, `0040-glm-parallel-deadlocks`, `0041-glm-parallel-ring-base`, `0048-glm-timing-tokens`, `0049-glm-multi-prefill`, `0065-glm-rank-checks` | several streams over one shared pool of per-token caches, one batched verify window a round, both ranks kept in step; prompts that arrive together filled in one forward (`MULTI_PREFILL`: 4 prose requests at once 103.4 -> 108.8 tok/s, first token 590 -> 340 ms); a request alone on the one-stream graphs (`TF_GLM_MULTI_LONE=1`, off by default since v1.3.1: +0.6-0.9%); the startup timings of verify windows on distinct tokens | 4 requests at once ([Performance](#performance)) |
| Sampling | `0034-cuda-nucleus-union` | a top_p draw from both ranks' candidates together, the same draw with fewer whole-shard reads (`TENSORFOLD_NUCLEUS_UNION=1`, off by default) | opt-in |
| Server | `0003-glm-vision`, `0050-glm-many-media`, `0056-glm-tool-result-media`, `0036-glm-tool-calls`, `0051-glm-tool-history-recovery`, `0053-glm-whole-tool-calls`, `0055-glm-open-tool-calls`, `0037-cuda-tokenize`, `0023-server-effort-max`, `0057-server-thinking-alias`, `0044-cuda-context-errors`, `0045-cuda-metrics`, `0058-server-client-gone-poll`, `0059-server-refused-bodies`, `0061-server-smooth-stream` | GLM's image and video processors and vision tower; up to 50 pictures and 4 clips a request in 96 MiB bodies, in user messages and tool results; GLM tool calls for agent clients; a past tool call whose arguments are not a JSON object left out of the prompt with its result and logged, instead of HTTP 400 (agents replay history, so a 400 ended the conversation); each call sent whole once written, a call the token limit cuts never sent, one the model ends without `</tool_call>` closed when it parses (else text), a missing `<arg_key>` put back; `/tokenize` and `/detokenize`; `reasoning_effort: "max"`; `chat_template_kwargs.thinking` read as `enable_thinking` (from Alexbob0's #25); the `param` field and GLM's own refusals on TensorFold v0.6.0's `context_length_exceeded` errors, and `/health`'s figures in its Prometheus `/metrics`; the client-gone check past 1,023 descriptors (TensorFold PR #218, jayleaton); a refused POST's body read before the reply (TensorFold v0.6.1's #181 fix); smooth streaming from a playout buffer (`STREAM_SMOOTH`) | the API features above |
| 3 Sparks (experimental) | `0066-glm-tp-n`, `0067-glm-tp3-split-pad`, `0068-glm-tpn-split-buffer-rows` | the engine on 2 or 3 ranks (`--tp`): heads, expert columns, vocabulary and DFlash2 KV groups split in whole units, the remainder to the lowest ranks; the row split (`SPLIT`) and its early connection to every peer; RoCE all-gathers over per-peer routes (b12x's proxy modified for more than two Sparks); prompt buffers and the memory estimate hold the split's pad row at three ranks | [3 Sparks](#3-sparks-experimental); two Sparks unchanged |
| Concurrent fill | `0062-glm-sliced-fill` | a new prompt's chunks run in layer slices (`FILL_BUDGET_MS`) with decode rounds between them, drafted as usual (`FILL_DRAFTS`), both ranks on the same layer boundaries; the chunk's buffers kept between slices, grouped fills, cache moves and cancellation mid-fill; whole forwards when nothing else decodes | with smooth streaming, the other replies' pauses during a 25k-token fill 630-690 ms -> 81-149 ms (table below) |

Concurrent prompt fill and smooth streaming on two Sparks (`PARALLEL=4`, DFlash2, 1,024-row chunks): three 400-token
replies already streaming when a fresh ~25k-token prompt arrives. Gaps are between one reply's events as the client
receives them, from that prompt's arrival to its first token; tokens = what the three replies received in that time.
One boot a row; the replies' text is the same in every row. The default is the last two rows.

| `FILL_BUDGET_MS` | `FILL_DRAFTS` | `STREAM_SMOOTH` | Gap p50 / p90 / max (ms) | Gaps > 250 ms | Tokens | New prompt's first token (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `0` (v1.3) | | `0` | 630 / 690 / 818 | 78 | 81 | 16.19 |
| `0` (v1.3) | | `0` | 632 / 673 / 754 | 75 | 81 | 16.07 |
| `150` | `0` | `0` | 185 / 199 / 230 | 0 | 324 | 19.49 |
| `150` | `0` | `0` | 185 / 199 / 211 | 0 | 327 | 19.49 |
| `200` | `0` | `0` | 220 / 244 / 256 | 9 | 243 | 17.85 |
| `200` | `0` | `0` | 229 / 245 / 270 | 12 | 246 | 17.99 |
| `300` | `0` | `0` | 310 / 334 / 342 | 123 | 174 | 16.71 |
| `300` | `0` | `0` | 314 / 335 / 338 | 126 | 171 | 16.74 |
| `200` | `0` | `1` | 201 / 236 / 250 | 1 | 288 | 18.12 |
| `300` | `0` | `1` | 305 / 335 / 344 | 111 | 202 | 16.56 |
| `300` | `1` | `1` | 106 / 201 / 379 | 18 | 448 | 18.94 |
| `300` | `1` | `1` | 101 / 200 / 392 | 17 | 450 | 18.89 |
| `200` | `1` | `1` | 81 / 150 / 230 | 0 | 650 | 21.13 |
| `200` | `1` | `1` | 81 / 149 / 231 | 0 | 662 | 21.32 |

Without a prompt filling, smooth streaming turns each round's burst into evenly spaced tokens: one reply's gaps p50 /
p90 50 / 68 -> 16 / 20 ms, four replies at once 107 / 128 -> 40 / 58 ms; the replies end when they did (last token
6.6 s alone, 16.8-17.0 s four at once, either way). A request alone fills as before: 50k and 150k-token prompts within
1%, 12k within the boot-to-boot drift (+0.25% against an unchanged build on the same day). The first boot of a new
image compiles a kernel during its first concurrent fill (one ~2 s pause, once).

## Checks

**Outputs.** Drafts only propose: every drafted token is checked against the model's own keyed sample, so drafted
replies equal TensorFold's serial, one-token-at-a-time decoding (send `"draft": false` for that reference). Replies
served 4 at a time equal the same requests served alone; `COMM=roce`, `SPLIT=1`, `TF_GLM_L2PF` and `TF_GLM_EXL3_LOADS`
move the same bits. Three defaults are not exact against the checkpoint in bf16, for speed and the 1M window:
`DENSE=q4` and `KV=fp8` are lossy (quality and the 1M needle: [Performance](#performance)), and `KDA_CHUNKED=1` is
close to the serial kernel but not its bits.
`DENSE=bf16 KV=bf16 KDA_CHUNKED=0` serves the checkpoint as it is, with a shorter window ([KV pool and memory](#kv-pool-and-memory)).

The checks in `tools/` talk to the running server (`API_URL`, default `http://127.0.0.1:8888`; or just `PORT`),
from the head or another machine (`API_URL=http://<head-address>:8888 tools/needle.py`). Performance is measured
with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) ([Performance](#performance)).

| Script | What it does |
| --- | --- |
| `tools/needle.py [label] [size]` | hides a passphrase in a ~195k-token prompt (the prompt comes out at ~0.8 x `size` tokens) and checks the model returns it |
| `tools/toolcheck.py` | makes a tool call with an array parameter and checks it comes back as a JSON array |
| `tools/end_of_turn.py [label] [max_cut]` | 8 short French coding prompts, thinking off: counts the replies that run to `max_tokens` (48 requests) and measures P(end of turn) right after each reply's closing code fence; exit 1 above `max_cut` cut replies (default 4) |
| `tools/prompt_reuse.py [size]` | a ~33k-token conversation takes three more turns, each after a request of another conversation with the same system prompt (an agent and its sub-agents); exit 1 when a turn resumes less than 90% of its prompt |

## Repository layout

```
start.sh      set up (first run) and start both ranks
start-tp3.sh  the same on three Sparks (experimental, patches 0066-0068)
stop.sh       stop them
scripts/      config.sh (all settings), local.sh.example (this setup's WORKER), prepare.sh (image + checkpoint on
              both Sparks), nodes.sh (ssh and the RoCE links), publish-image.sh (push the image to GHCR),
              banner.sh (start.sh's banner)
patches/      patches baked into the image
tools/        checks against the running server (needle, tool calls, end of turn, prompt reuse)
CHANGELOG.md  what changed in each release
CREDITS.md    who and what this builds on
LICENSE       Apache License 2.0
NOTICE        third-party notices (TensorFold's MIT and Apache-2.0 notices, b12x, glm53-tensorfold-spark, ShapleyMcg)
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE). [`NOTICE`](NOTICE) carries the third-party notices that go with it: the
files in `patches/` modify TensorFold v0.6.0, and the TensorFold code they change or quote as context stays under
TensorFold's licenses (Apache 2.0 from v0.6.0, and the MIT notice of code written before it, both in `NOTICE`); parts
of patches 0006 (b12x), 0036, 0046 and 0047 (glm53-tensorfold-spark) come from Apache-2.0 projects, credited there and
in [`CREDITS.md`](CREDITS.md). The model
files are downloaded from Hugging Face and are not part of this repository:

- **The checkpoint** [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) is under the Apache License 2.0; the base model it derives from is MIT-licensed by Z.AI.
- **The base model** [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is under the license on its model
  card.
- **The DFlash2 drafter** is under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/),
  non-commercial use only (commercial licensing: contact@inco.ai); `DRAFTER=mtp` serves without it.

**Third-party software in the image.** The prebuilt image (and the one `scripts/prepare.sh` builds) is based on
NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA
software in it is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start; by pulling or running the image you accept them. The image also contains
PyAV (BSD) with its FFmpeg libraries (LGPL) and xgrammar (Apache 2.0). The Apache License above covers this
repository's own work only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)),
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai, the EXL3 format and converter (exllamav3) by
turboderp, the
DFlash2 drafter by
[IncoAI](https://huggingface.co/incoai), b12x's RoCE transport by local-inference-lab, and code from
[glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) by Jay Leaton (tool calling, L2 prefetch,
expert loads). The full list, including the runtime stack and licenses, is in [`CREDITS.md`](CREDITS.md).
