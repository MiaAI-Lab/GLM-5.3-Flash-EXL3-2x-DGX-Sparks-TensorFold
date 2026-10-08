# Credits

This repository is a thin layer of scripts and patches. Almost everything that makes it work was built by others.
Its own work is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the
third-party notices that go with it (TensorFold's MIT and Apache-2.0 notices, b12x, glm53-tensorfold-spark, the
checkpoint's ShapleyMcg attribution and the drafter's license).

## Model

- **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** by [Z.ai](https://huggingface.co/zai-org): the
  model's design, training and evaluations. Its license, on the model card, governs any use of the weights. The
  weights are not part of this repository; `scripts/prepare.sh` downloads them from Hugging Face.
- **Mia's AI Lab**: the default checkpoint since v1.3.3, [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) (Apache-2.0), made with
  [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- **[brandonmusic](https://huggingface.co/brandonmusic)**: the default checkpoint before v1.3.3, still selectable
  with `MODEL_ID`: the EXL3 4-bit quantization,
  [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw), served here
  from its mirror [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
  (a byte-identical copy), made with **ShapleyMcg** by Brandon M.
  Music under the **ShapleyMcg License 1.0** (attribution required; the checkpoint's `LICENSE` file has the terms).
  Its attribution notice:

  > This work includes or was produced using ShapleyMcg, created by Brandon M. Music
  > (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an
  > attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without
  > this attribution is unlicensed.
- **[IncoAI](https://huggingface.co/incoai)**: the DFlash2 drafter,
  [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (**CC BY-NC-ND 4.0**:
  non-commercial use, no derivatives). Downloaded from its source, never redistributed here.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache 2.0 from v0.6.0; releases up to v0.5.0 were MIT, and code written before v0.6.0
  keeps its MIT notice): the engine that serves the model, including the two-rank CUDA engine for GLM-5.3-Flash, its
  EXL3 expert kernels, DFlash2 and MTP drafting with exact verification and the OpenAI-compatible server. Every file
  in `patches/` is a modification of TensorFold v0.6.0.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v0.6.0/THIRD_PARTY_NOTICES.md):
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), whose EXL3 format the routed experts are
  stored in, the GLM-5.3-Flash modeling code in Hugging Face
  [transformers](https://github.com/huggingface/transformers) (Apache 2.0), and
  [z-lab/dflash](https://github.com/z-lab/dflash) (Z Lab, MIT), whose DFlash2 architecture its drafter ports.

## Patches

- `0003-glm-vision`: GLM's image and video processors (resize with pad, 2 fps frame choice, prompt layout) and vision
  tower, checked bit for bit against Hugging Face [transformers](https://github.com/huggingface/transformers) 5.17
  (Apache 2.0), the reference they follow; builds on TensorFold's Qwen image pipeline.
- `0006-cuda-roce-allgather`: the one-shot RoCE all-gather (`COMM=roce`) is the "RoCEnante" transport of
  **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab (Apache 2.0): its C proxy
  (`roce_proxy.c`), modified for more than two Sparks (per-peer routes over up to 4 network cards; its header lists the
  changes), and its CuTe all-gather kernel reimplemented in CUDA C++ (`roce.cu`).
- `0012-glm-kda-chunked`, `0014-glm-kda-chunked-gb10`, `0039-glm-kda-chunked-kernel` (`KDA_CHUNKED`, on by default):
  the chunked WY / UT form of the delta-rule recurrence follows the published chunkwise algorithm of gated delta
  networks and Kimi Delta Attention, as implemented in
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT); the kernels are written anew.
- `0013-glm-decode-rounds`: verify windows of up to 16 rows so copy drafts can run long, after the wider copy windows
  proposed in [TensorFold PR #115](https://github.com/ashhart/TensorFold/pull/115) by Ash Hart.
- `0036-glm-tool-calls` (tool calling for agent clients): the rule that tool calls written inside the think block
  count when they end the reply, the store that gives back a step's reasoning to clients that drop it, and null /
  `const` typing of tool arguments are adapted from patch 0620 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); the rest of
  that patch's fixes are our own implementation.
- `0037-cuda-tokenize`: the `/tokenize` and `/detokenize` endpoints take and return what
  [vLLM](https://github.com/vllm-project/vllm)'s (Apache 2.0) do, so clients written for vLLM work unchanged;
  the code is our own.
- `0043-glm-visible-pools`: the idea of bounding decode token selection to the pools a row can see follows
  [TensorFold PR #140](https://github.com/ashhart/TensorFold/pull/140) by mikolaj92, which TensorFold v0.6.0 has for
  its selection kernel; this patch's bound on the decode scoring and the split selection is our own.
- `0044-cuda-context-errors`: TensorFold v0.6.0's context-window refusals (commit 68c6e35 by Ash Hart) extended: the
  `param` field, and the same wording and `context_length_exceeded` code for the GLM app's window check and image
  prompts.
- `0045-cuda-metrics`: TensorFold v0.6.0's Prometheus `/metrics` (commit 4447ac3, #110, by Ash Hart) extended with
  `/health`'s figures as `tensorfold_health:` metrics.
- `0046-glm-l2-prefetch` (L2 prefetch in decode, `TF_GLM_L2PF`): adapted from patch 0460 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); changes listed
  in `NOTICE`.
- `0047-glm-exl3-decode-loads` (routed-expert decode loads, `TF_GLM_EXL3_LOADS`): adapted from patch 0580 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); changes listed
  in `NOTICE`.
- `0054-glm-image-prompt-reuse` (image prompts resume from kept prompt states, issue #11): by
  [abhicnv007](https://github.com/abhicnv007), applied as contributed, with two small review changes.
- `0057-server-thinking-alias` (`chat_template_kwargs.thinking`): from [Alexbob0](https://github.com/Alexbob0)'s
  pull request #25; the `{"type": ...}` forms and the refusal of other values were added here.
- `0058-server-client-gone-poll`: [TensorFold PR #218](https://github.com/ashhart/TensorFold/pull/218) by
  [jayleaton](https://github.com/jayleaton) (Apache 2.0), applied unchanged: the client-gone check sees descriptors
  past 1023.
- `0059-server-refused-bodies`: TensorFold v0.6.1's fix for #181 (commit 50dfe38a, by
  [SxMShaDoW](https://github.com/SxMShaDoW)), backported to v0.6.0.
- `0060-glm-keep-thinking` (earlier turns keep their reasoning, `TF_GLM_CLEAR_THINKING`): by
  [kky42](https://github.com/kky42), pull request #23.
- `0071-glm-shared-prefix-copy` (a shared system prompt's resume keeps the writer's kept prompt, issue #43): by
  [ezoushen](https://github.com/ezoushen), [pull request #44](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/44),
  applied as contributed, with `tools/prompt_reuse.py`.
- `0072-glm-display-kv` (`DISPLAY_KV_MIB`, the display reservation in the shared pool, issue #55): by
  [ezoushen](https://github.com/ezoushen), [pull request #56](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/56),
  applied as contributed, with `tools/display_kv_check.py` (one review fix in its `start.sh` check, which ended a headless
  Spark's start without a message). It applies to TensorFold's pool the display-reserve KV
  technique that [gabewillen](https://github.com/gabewillen) proposed for the vLLM kit in
  [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks#234](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/234).
- `0073-glm-queued-cancellation` (requests waiting for a slot are dropped when their client leaves): by
  [desy0305](https://github.com/desy0305), [pull request #51](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/51),
  applied as contributed, with `tools/test_queued_cancellation.py`; its delivery-failure handling comes from
  [johnwhited](https://github.com/johnwhited)'s [pull request #48](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/48).
- `0074-glm-compact-before-evict` (the shared pool compacts before it evicts, issue #61): by
  [ezoushen](https://github.com/ezoushen), [pull request #62](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/62)
  (contributed as 0073), applied as contributed, with `tools/pool_pressure.py` and `tools/pool_room_check.py`.
- `0077-glm-kept-cap-shared-by-recency` (a new agent run keeps its system prompt's state once the kept-state cap is
  full): the diagnosis, and the fix of evicting shared-prefix states by recency only, by
  [meleesciony](https://github.com/meleesciony) in [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75); the code is our own.
- `0078-glm-take-over-decide-then-copy` (a fresh conversation after a long one clones only the kept states that stay):
  by [m-naoki-m](https://github.com/m-naoki-m), also proposed upstream as
  [TensorFold PR #421](https://github.com/ashhart/TensorFold/pull/421).
- `0079-glm-picture-cache` and `0080-vision-quoted-markers`: by [ThomasWadeZ](https://github.com/ThomasWadeZ)
  (#63, #64).
- `0081-tfcap-capacity-status`, `0082-tfcap-admission-cap` and `0083-tfcap-delivery-abort`: by
  [johnwhited](https://github.com/johnwhited) (#48); the admission check and the abort-on-delivery-failure semantics
  follow [vLLM](https://github.com/vllm-project/vllm) v1's `check_admission` and generate-abort behaviour
  (Apache 2.0) as a reference.
- `0084-glm-spill-tier` (kept prompt states on local disk, `SPILL_GIB`): by [wojo](https://github.com/wojo). The
  idea of persisting session state on NVMe per rank comes from MiaAI-Lab's GLM-5.3-Flash vLLM kit
  ([pull request #232](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/232) by
  [gabewillen](https://github.com/gabewillen); the idea only, no code from that AGPL-3.0 project); the multi-rank
  design follows patch 0250 of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache 2.0); the file
  conventions and flag names follow TensorFold's MLX spill ([PR #68](https://github.com/ashhart/TensorFold/pull/68) by
  [gilby](https://github.com/gilby)) and [issue #155](https://github.com/ashhart/TensorFold/issues/155) by
  [raymondkpwong](https://github.com/raymondkpwong); which kept
  prompts leave first is `0063-glm-kept-cap-superseded-first` (PR #32 by [Alexbob0](https://github.com/Alexbob0));
  checking every file on read, private file modes, a per-Spark weights fingerprint and naming a rank whose settings
  differ follow the session tier of [JSpark3](https://github.com/jakejharris/jspark3) v2.0.1 by
  [jakejharris](https://github.com/jakejharris). The code is new.
- Every patch, except the parts credited above: by MiaAI-Lab, developed with
  [Claude Code](https://claude.com/claude-code), under the Apache License 2.0; the TensorFold code the patches modify or
  quote as context stays under TensorFold's licenses (Apache 2.0, and MIT for code written before v0.6.0; see
  `NOTICE`).

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA, cuDNN, cuBLAS, NCCL and related
  libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products.
- **[PyTorch](https://pytorch.org/)** (BSD-3-Clause): tensors, CUDA streams and the C++ extension builder that compiles
  the patches' CUDA kernels.
- **[Triton](https://github.com/triton-lang/triton)** (MIT): the language many of TensorFold's CUDA kernels are
  written in.
- **[NCCL](https://github.com/NVIDIA/nccl)** (BSD-3-Clause) and **[rdma-core](https://github.com/linux-rdma/rdma-core)**
  (libibverbs, GPL-2.0 / BSD-2-Clause): the two ranks' exchanges over the Sparks' ConnectX-7 RoCE link.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) and **[FFmpeg](https://ffmpeg.org/)** (LGPL): video
  decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding.
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (Apache 2.0): structured outputs (`response_format` and the
  `guided_*` fields).
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory), two of them linked by their ConnectX-7 ports: every number in the README was measured there.

If you believe something here is missing or credited wrongly, please open an issue.
