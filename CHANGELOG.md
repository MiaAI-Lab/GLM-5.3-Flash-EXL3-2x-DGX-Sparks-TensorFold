# Changelog

Every change to this recipe, newest first. Each release names the image it serves: `scripts/prepare.sh` pulls
`ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` by the digest pinned in `scripts/config.sh`.

## Unreleased

### Added
- `ABLIT=1`: serve the weights-level abliterated checkpoint
  [`bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit`](https://huggingface.co/bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit) @
  `14858211ed81d7fa773f8a0db02f38f36d230252` instead of the stock TR3 4-bpw pack. The runtime is checkpoint-agnostic:
  the abliteration is baked into those weights (the checkpoint's `ABLIT_META.json` is data the loader ignores), so no
  other setting changes. `scripts/prepare.sh` downloads exactly that revision and copies it to the worker like any
  model switch; an explicit `MODEL_ID` still wins over `ABLIT`. No image change (script-level setting only).

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
