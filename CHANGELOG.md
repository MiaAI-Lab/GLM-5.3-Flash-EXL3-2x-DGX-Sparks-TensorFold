# Changelog

Every change to this recipe, newest first. Each release names the image it serves: `scripts/prepare.sh` pulls
`ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` by the digest pinned in `scripts/config.sh`.

## Unreleased

### Changed
- **Earlier turns' reasoning stays in the prompt** (patch `0054-glm-keep-thinking`), as in zai-org's current template.
  The checkpoint's template dropped it at each new user message, so agents prefilled the previous turn's tool loop
  again (a replayed agent session: 85,127 -> 16,106 tokens). `TF_GLM_CLEAR_THINKING=1` restores the old rendering.

## v1.3.2 (2026-10-01): more kept prompts, a note on non-English prompts

Image unchanged: `v0.6.0-ae8d1c789b47`.

### Changed
- **`TF_GLM_CACHE_ENTRIES` 8 -> 32** (#17): the engine kept at most 8 prompt states and dropped the oldest past that,
  however much of the pool was free. An agent request keeps 1 to 3 (its own state plus shared-prefix states), so three
  or four alternating agent conversations pushed each other out. Each entry reserves ~45 MiB at start, ~1 GiB more in
  all. Reported, with measurements, by @sm373373. The setting is now documented.

### Docs
- **`DENSE=q4` and non-English prompts** (#18): on short French coding prompts `q4` often loses the end of turn and
  runs to `max_tokens` (12 of 48 replies, against 1 of 48 with `bf16` or `fp8`); the README says so and points to
  `DENSE=fp8`. A calibrated 4-bit for the dense weights, at `q4`'s speed, is being worked on. Reported, with
  measurements, by @Alexbob0.
- `DENSE=q4` keeps `kv_b` in BF16 (the docs said FP8).

## v1.3.1 (2026-10-01): prompt cache in interactive sessions, setup fixes

Image unchanged: `v0.6.0-ae8d1c789b47`.

### Changed
- **`TF_GLM_MULTI_LONE` now defaults to 0** (#12, #13). With 1, a request decoding alone moved to the pool's first rows
  for the one-stream graphs and evicted the other conversations' kept prompts there, so conversations taking turns
  re-read their whole histories. Measured with three conversations taking turns (~4.5k tokens each, 6 warm turns):
  0% of the prompts came from the cache with 1, 98.1% with 0. The one-stream graphs gave +0.6-0.9% at 1 stream; `1`
  still turns them on. Reported by @abhicnv007 and @kky42.

### Fixed
- **#9, `prepare.sh` picked the LAN port when `WORKER` is a LAN address.** When the route to the worker leaves through
  a port without RoCE and `FABRIC_PEER` is unset, the worker's CX7 address that a local CX7 port reaches directly is
  now found and used (logged). Reported by @ttnghia.
- **#15, the weight copy to the worker missed files downloaded with huggingface_hub's xet backend:** their blobs are
  links into the cache root's `blobs/`, outside the copied folder. The copy now follows those links. Reported by
  @huitseeker.
- **#8, `prepare.sh` failed on every run when the two Sparks use different Docker image stores** (containerd on one,
  overlay2 on the other): it compared image `.Id`s, which is the manifest digest under containerd and the config
  digest under overlay2, so the same image never matched. Images are now compared by content (their layers' diffIDs
  and runtime config), the same under both stores. Fix by @eleata, confirmed by @kafej; also reported in #14 by
  @huitseeker.

## v1.3 (2026-10-01): TensorFold v0.6.0, issue fixes #2 and #6, whole tool calls

Image `v0.6.0-ae8d1c789b47` (`sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969`), 53 patches.

### Changed
- **TensorFold v0.6.0** (was v0.5.0). The patches are rebased onto it. Three of ours are now part of TensorFold
  itself and were dropped: the idle doorbell, `TENSORFOLD_MEMORY_RESERVE_GIB` and `TF_GLM_MTP`. The others keep
  their names, renumbered (0024 -> 0023 ... 0053 -> 0050; the README's patch table lists them). TensorFold is
  Apache-2.0 from v0.6.0; `NOTICE` and `CREDITS.md` say so.
- `TF_GLM_MTP=auto` is now set explicitly: v0.6.0's own default (`1`) would load the MTP head (1.77 GiB a Spark)
  with `PARALLEL=1`.
- **Default reply budget 4,096 -> 32,768 tokens** (`MAX_TOKENS`) for a request that sets no `max_tokens`, as agents
  such as Codex do. GLM thinks at Max by default and could run out of budget inside a tool call. A request's own
  value still wins; the default is cut to what the window has left, never refused.
- **Compiled kernels are kept per image** (`~/.cache/tensorfold-glm53/<image hash>`): another image's build of the
  same extension can no longer be loaded by mistake. A new image compiles once (a few minutes on its first start).
- From v0.6.0 itself: `reasoning_effort: "medium"` is heard as `high`; `logprobs`, `top_logprobs` and `n` other than
  1 get HTTP 400 (they were ignored).

### Fixed
- **#6, `COMM=roce` first start dying at the first all-gather.** On a first start each rank builds its CUDA kernels on
  its own and could drift past the ~20 s a RoCE wait allows. Until startup is over, each RoCE gather first waits for
  the peer in an NCCL barrier; the RoCE kernel builds during setup. New `TF_ROCE_WAIT_S` (default 20 s); a failure
  now prints the proxy's counters. Patch `0052-cuda-roce-startup`.
- **#2, one malformed tool call in the history blocking a conversation for good.** A past tool call whose arguments
  are not a JSON object is left out of the prompt with its result, and logged, instead of HTTP 400 on every later
  turn. Patch `0051-glm-tool-history-recovery`.
- **Cut-off tool calls are never sent.** A tool call now goes out whole once it is written; a reply that ends inside
  a call (its token limit) ends with `length` and never sends that call, streamed or not, so a client cannot store
  or run cut arguments. While a call is written, an empty delta goes out every 2 s for clients with an idle timeout.
  Patch `0053-glm-whole-tool-calls`.

### Unchanged
- Replies: the exactness checks (reference shas, drafted == serial 6/6, concurrency 22/22, images and videos, tool
  calls) equal v1.2's. Prefill speed is unchanged (sparkDash, 8k-256k).
- Memory: KV pool **2,922,496 tokens** at the measured start (the 12.5 GiB cap); lowest free memory under a 1M-token
  prompt 5.6 GiB on the head, 9.5 GiB on the worker (needle found).

## v1.2 (2026-10-01): bigger KV pool, worker weights over NFS

Image unchanged: `v0.5.0-cb7c56f7f921` (`sha256:6ee3c6e0430040b69ddcb0c96c7fbbcb94a5bed47d48a8ba092626369ae533b9`),
53 patches.

### Added
- `WORKER_WEIGHTS=nfs`: the worker keeps no copy of the checkpoint and DFlash2 (~166 GiB less disk on it). Rank 1
  reads the head's Hugging Face cache read-only over NFS, through a docker volume that `prepare.sh` creates on the
  worker (no sudo there), after checking the worker sees every file of both snapshots as the head has them. The head
  exports its cache once (README: Worker weights over NFS). Settings `NFS_PATH`, `NFS_SERVER`, `NFS_VOLUME`. The
  default stays `copy`. Measured: both ranks live in ~2.2 minutes, as with a local copy.

### Changed
- `KV_POOL_GIB` 11 -> **12.5**: the shared KV pool is **2,852,864 tokens** at the measured start (was 2,684,928),
  ~2.1-2.9M depending on what is free at start. Lowest free memory under a 1M-token prompt: 4.7 GiB on the head,
  8.8 GiB on the worker (was 6.3 / 10.8); idle 7.8 / 10.8 GiB.
- `prepare.sh` and `start.sh` no longer show TensorFold's "EXL3 support is experimental" note: this recipe serves
  the EXL3 checkpoint on purpose, and its replies are checked exact.

### Removed
- `tools/bench.py`. The recipe's performance numbers come from
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash), measured through the OpenAI API from another machine, so a
  second benchmark in the repo only gave numbers that did not match the published ones. Its shared helpers (the
  server URL, error messages, random prose) moved to `tools/client.py`; `tools/needle.py` and `tools/toolcheck.py`
  stay as correctness checks.

### Unchanged
- Replies: the exactness checks (reference shas, drafted == serial 6/6, concurrency 22/22, images and videos, tool
  calls) equal v1.1's, and the 1M-token needle is found (981,841 tokens, 967 s).

## v1.1 (2026-10-01): up to 50 images and 4 videos a request

Image `v0.5.0-cb7c56f7f921` (`sha256:6ee3c6e0430040b69ddcb0c96c7fbbcb94a5bed47d48a8ba092626369ae533b9`), 53 patches.

### Added
- Patch `0053-glm-many-media`: up to **50 images** and **4 videos** a request (was 4 and 2).
  - A request's images share 16,384 tokens: up to 8 keep the full 2,048 each, more get an equal share (327 with 50).
    Its videos share 32,768: 1 or 2 keep 16,384 each, 3 or 4 get 10,922 or 8,192.
  - Request bodies up to 96 MiB (was 32 MiB), so data URLs can carry about 70 MB of images and videos; images up to
    64 MB in all (was 20 MB).
  - Each image is resized as it is decoded, so only one full-size image is in memory at a time, and the vision
    tower's output is no longer copied once more at the end: lower peak memory for every image or video request.
  - New settings: `TENSORFOLD_GLM_MAX_IMAGES` (50), `TENSORFOLD_GLM_MAX_VIDEOS` (4),
    `TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS` (16,384), `TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS` (32,768).
- `CHANGELOG.md`.

### Changed
- The published image is pinned by digest (`IMAGE_TAG` / `IMAGE_DIGEST` in `scripts/config.sh`): `prepare.sh` pulls
  exactly that image while the patches are this release's. `scripts/publish-image.sh` prints the values to pin after
  a push; `start.sh` shows the patches hash it serves.

### Unchanged
- Replies to text requests, and to requests with up to 4 images and 2 videos, are byte-identical to v1.0 (drafted
  and serial); the KV pool (2,684,928 tokens) and speed are the same.
- Measured: 50 images in one request answered in 12.5 s (16,239 prompt tokens), lowest free memory on the head
  6.75 GiB; 4 videos 6.90 GiB.

## v1.0 (2026-10-01): first release

Image `v0.5.0-cefe8bf45d07` (`sha256:7bcbb617b1f40f1ce12d37e5ba5e42caf1b1c53b444be9951915e966314c3e1d`), 52 patches.

- GLM-5.3-Flash (EXL3 4-bit routed experts, checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`) on two DGX Sparks
  with TensorFold v0.5.0 and 52 patches, as an OpenAI-compatible API on port 8888 (model id `GLM-5.3-Flash-EXL3`).
- 1,048,576-token context, up to 4 requests at once over one shared FP8 KV pool of ~2.7M tokens, prompts that
  arrive together filled in one forward.
- DFlash2 and copy drafts (drafted replies equal serial ones), 4-bit dense weights, the RoCE all-gather between the
  Sparks, prompt and shared-system-prompt reuse.
- Image and video input (up to 4 images and 2 videos a request), tool calling, structured outputs, `/tokenize`,
  `/metrics`, `context_length_exceeded` refusals.
- `start.sh` sets everything up on first run (image, pinned checkpoint and drafter, copy to the worker) and starts
  both ranks; `stop.sh` stops them.
- Measured with sparkDash: one request 60.4 tok/s prose, 114.7 structured; 4 at once 108.8 / 227.9 tok/s in all;
  prefill ~1,950 tok/s up to 64k tokens.
