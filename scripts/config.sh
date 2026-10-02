# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# The two Sparks: this machine serves rank 0 and the API; WORKER (ssh target, key-based) runs rank 1.
WORKER="${WORKER:-}"                 # e.g. user@<worker address>; set it in scripts/local.sh
FABRIC_PEER="${FABRIC_PEER:-}"       # the worker's CX7 address when WORKER is reached over another network
MASTER_PORT="${MASTER_PORT:-29551}"  # TensorFold's rendezvous port between the ranks (keep it on the private link)

MODEL_ID="${MODEL_ID:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"   # EXL3 routed experts (4 bpw), BF16 elsewhere
# The checkpoint's revision (a Hugging Face commit sha; DFLASH2_REVISION below is DFlash2's): the one this recipe was
# measured with. prepare.sh downloads exactly it, start.sh serves that snapshot from the local cache (no network), and
# a new upstream commit changes nothing here until the pin does. Empty: the Hub's main when first downloaded. The pin
# belongs to the default MODEL_ID; another MODEL_ID gets no pin unless you set one.
_rev=""; [[ "$MODEL_ID" == Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw ]] && _rev=9eaebb7c4e96d983dcd538e18624622ba5b820a8
MODEL_REVISION="${MODEL_REVISION-$_rev}"
TF_VERSION="${TF_VERSION:-v0.6.0}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"
# pip packages the image adds on top of TensorFold (av: video input; xgrammar: response_format / structured outputs);
# they are part of the image's hash, so a change rebuilds it like a patch does
IMAGE_EXTRAS="av==18.1.0 xgrammar>=0.2.8,<0.3"
image_hash() { (cat patches/*.patch 2>/dev/null; echo "$IMAGE_EXTRAS") | sha256sum | cut -c1-12; }
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold}"
# The published image of this release's patches, pinned: prepare.sh pulls it by digest (a tag can be moved, a digest
# cannot) while patches/*.patch and IMAGE_EXTRAS still hash to IMAGE_TAG's hash. Other patches pull
# $GHCR_IMAGE:<TF_VERSION>-<hash> when one is published, else build locally. scripts/publish-image.sh prints both.
IMAGE_TAG="${IMAGE_TAG:-v0.6.0-ae8d1c789b47}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969}"
# the registry reference prepare.sh pulls for these patches: the pinned digest, or the hash's tag
prebuilt_image() {
  local tag="${TF_VERSION}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-tf}"           # the same name on both Sparks

SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
DRAFTER="${DRAFTER:-dflash2}"        # dflash2: incoai/GLM-5.3-Flash-DFlash2 drafts (CC BY-NC-ND 4.0: non-commercial
                                     # use only), +5-10% decode over mtp; mtp: the checkpoint's own MTP head
# The checkpoint's MTP head beside DFlash2 (TensorFold's TF_GLM_MTP): auto (default) leaves it out while DFlash2
# drafts every request; TensorFold v0.6.0's own default, 1, would load it (1.77 GiB a Spark) with PARALLEL=1.
export TF_GLM_MTP="${TF_GLM_MTP:-auto}"
# Image and video input (rank 0 runs GLM's vision tower: 1.05 GiB of bf16 weights and 0.75 GiB of workspace). A picture
# takes at most TENSORFOLD_GLM_IMAGE_TOKENS tokens (2048), a clip TENSORFOLD_GLM_VIDEO_TOKENS (16384) over at most
# TENSORFOLD_GLM_VIDEO_FRAMES frames (128, 2 a second). VISION_URLS=1 also accepts public https URLs (default: data URLs).
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
# Concurrent requests (patches 0026-0030, 0035, 0040, 0041: one shared pool of per-token caches, one batched verify window
# a round): 1 to 4, with DRAFTER=dflash2 only (mtp: 1). 4 (default), prose in all (sparkDash): 60.4 / 79.2 / 89.5 /
# 108.8 tok/s at 1 / 2 / 3 / 4 at once; structured 114.7 / 147.6 / 196.3 / 227.9.
if [[ "$DRAFTER" == dflash2 ]]; then _par=4; else _par=1; fi
PARALLEL="${PARALLEL:-$_par}"
# The DSA latent cache and the indexer's pooled keys (patch 0038): fp8 (default) holds them as e4m3 rows with a
# power-of-two scale each, half bf16's bytes: the 1M-token window with 4 streams fits (rank 0: 88.09 GiB estimated,
# pool 2,922,496 tokens at the measured start). Lossy: GSM8K 98.0%, HumanEval 97.6%, 1M needle found; drafted replies still equal serial
# ones. bf16: the exact cache (~196k tokens with DFlash2).
KV="${KV:-fp8}"
export TF_GLM_KV="$KV"
# Prompt + reply window. With KV=fp8: 1,048,576 (the checkpoint's native window). With KV=bf16: 196,608 with DFlash2
# (163,840 with VISION=1 and DENSE=fp8 or bf16: the tower costs rank 0 ~66k tokens of window, which q4's smaller
# weights give back), 524,288 with mtp. start.sh falls back to the largest that fits when a start's memory budget is
# smaller. 0: the largest the memory affords (then no memory is left to keep other conversations' prompts).
DENSE="${DENSE:-q4}"
if [[ "$KV" != bf16 ]]; then _ctx=1048576; elif [[ "$DRAFTER" != dflash2 ]]; then _ctx=524288
elif [[ "$VISION" == 1 && "$DENSE" != q4 ]]; then _ctx=163840; else _ctx=196608; fi
CONTEXT="${CONTEXT:-$_ctx}"
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"
_rev=""; [[ "$DFLASH2_ID" == incoai/GLM-5.3-Flash-DFlash2 ]] && _rev=bf582e4eacc1810f76656d1811693ff6c6737d2a
DFLASH2_REVISION="${DFLASH2_REVISION-$_rev}"   # DFlash2's pinned revision, as MODEL_REVISION above
THINKING="${THINKING:-1}"
# The reply budget of a request that sets no max_tokens (or max_completion_tokens), reasoning and answer together:
# 32768. GLM thinks at Max by default, and TensorFold's own 4,096 could end a reply inside a tool call (an agent such
# as Codex sets none). A request's own value wins; this one is cut to what the window has left, never refused.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# The checkpoint's BF16 weights (attention, shared experts, dense layers, head: ~9.7 GiB a Spark):
#   q4 (default): the projections as affine 4-bit groups of 64 with MSE-searched ranges, the head in FP8 (kv_b stays BF16: TF_GLM_KVB)
#        (patches 0002, 0005). Over fp8 (DFlash2, one boot each): prose 38.9 -> 44.4 tok/s, code 44.2 -> 48.7, prefill
#        ~1,090 -> ~1,260 tok/s; GSM8K 98.0% and HumanEval 95.1% on both (bf16: 97.2 / 96.3). Lossy: replies differ.
#   fp8: FP8 e4m3 with a scale per row and 128 columns (patch 0002), half bf16's bytes: ~+33% decode over bf16.
#   bf16: the checkpoint as it is.
export TF_GLM_DENSE="$DENSE"
# The ranks' all-gathers. roce (default): the small ones (a decode round's partials, the samplers; up to
# TF_ROCE_MAX_KB below) as one-shot RDMA writes over the Sparks' RoCE link, b12x's transport (patch 0006): 11 us a
# 16 KiB gather against NCCL's 45; decode +6% (prose 44.2 -> 46.9, code 48.7 -> 51.6). NCCL keeps the rest. nccl: NCCL
# for all. Same bits.
COMM="${COMM:-roce}"
export TF_GLM_COMM="$COMM"
# The largest all-gather in KiB that goes over RoCE (patch 0006 reads it; a setting, no patch of its own): 512 (default;
# TensorFold's is 256) also takes the 17-32-row verify windows of concurrent requests: code at 4 streams +1.4%, prose
# +0.5-1% (two boots each). Same bits.
export TF_ROCE_MAX_KB="${TF_ROCE_MAX_KB:-512}"
# Prompt-lookup ("copy") drafts (patch 0007): when the reply's last 8 tokens occurred before, the tokens that followed
# them are verified ahead of DFlash2's; quote / edit replies +5% (80.3 -> 84.4 tok/s), prose and code unchanged. Exact.
COPY="${COPY:-1}"
export TF_GLM_COPY_DRAFTS="$COPY"
# Copy drafts a round (patch 0013: verify windows up to 16 rows): 15 lets a long quote go through in one round; edit
# replies 84 -> 119 tok/s, prose and code unchanged (two boots each). Exact.
COPY_MAX="${COPY_MAX:-15}"
export TF_GLM_COPY_MAX="$COPY_MAX"
# How many DFlash2 drafts a round verifies (patch 0018): fnc7:0.3 stops a chain when the product of each draft's chance
# under the request's own keyed sampling noise drops below 0.3 (up to 7 drafts); prose 47.2 -> 50.6 tok/s, code 51.7 ->
# 55.5 over fc5:0.3 (two boots each). Drafts only propose: replies are the same under every policy.
DRAFT_POLICY="${DRAFT_POLICY:-fnc7:0.3}"
export TF_GLM_DFLASH_POLICY="$DRAFT_POLICY"
# Prompt chunks' hyper-connection glue split by rows between the two Sparks, its exchanges overlapped with the next
# rows' work (patch 0010): prefill ~1,270 -> ~1,730 tok/s on a 50k prompt (with patches 0009 and 0020); decode rounds
# pay ~1.5%. The overlap also runs the next block's front on its own rows during the exchanges (patch 0033; with
# COPY_CODE below, two boots each: a 149k prompt 92.8 -> 90.7 s). Same bits. SPLIT=0 turns it off.
SPLIT="${SPLIT:-1}"
export TF_GLM_HC_SPLIT="$SPLIT" TF_GLM_PREFILL_OVERLAP="$([[ "$SPLIT" == 1 ]] && echo 2 || echo 0)"
# KDA prompt chunks in chunked (WY) form, one CUDA kernel of 32-row sub-chunks (patches 0012, 0014, 0039): prefill
# 50k 29.3 -> 26.4 s, 149k 91.1 -> 84.5 s (one boot each); prompt states then sit on a 64-token grid.
# Close to the serial kernel, not its bits: prompt arithmetic differs, drafted replies still equal serial ones. 0: off.
KDA_CHUNKED="${KDA_CHUNKED:-1}"
export TF_GLM_KDA_CHUNKED="$KDA_CHUNKED"
# Code-workload copy drafts (patch 0032): 16-row verify windows as CUDA graphs, and a copy from the reply itself only
# when its last 16 tokens match: code 55.4 -> 56.2 tok/s, edit 120 -> 125 (with the overlap above, two boots each).
# Exact. 0: off.
COPY_CODE="${COPY_CODE:-1}"
_w=0; [[ "$COPY_CODE" == 1 ]] && _w=16
export TF_GLM_WIDE_GRAPHS="$_w" TF_GLM_COPY_REPLY_MATCH="$_w"
# --parallel: a request alone runs on the one-stream graphs (patch 0035; 1) instead of the batched ones (0, default):
# +0.6-0.9% at 1 stream, but to use them the stream moves to the pool's first rows and evicts the other conversations'
# kept prompts there, so interactive sessions miss the prompt cache and re-read whole histories (issues #12, #13).
# Off by default until that move keeps them. Exact either way.
export TF_GLM_MULTI_LONE="${TF_GLM_MULTI_LONE:-0}"
# Kept prompt states (TensorFold's TF_GLM_CACHE_ENTRIES, 8 by default): the oldest is dropped past this count, however
# much of the pool is free. A drafted agent request keeps 1 to 3 (its own plus shared-prefix states), so 8 let three
# or four alternating conversations push each other out (issue #17). Each entry reserves its fixed state (~45 MiB) at
# start: 32 takes ~1 GiB more than 8.
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-32}"
# Earlier turns keep their reasoning in the prompt (patch 0054), as in zai-org's current template. 1: drop it, as the
# checkpoint's template does; agents then prefill the previous turn's tool loop again at each new user message.
export TF_GLM_CLEAR_THINKING="${TF_GLM_CLEAR_THINKING:-0}"
# Waiting prompts filled together in one forward (patch 0049): shared work (expert weights, glue, projections) runs once
# for every waiting prompt, attention per prompt on its own state, so each gets the bits it gets alone. sparkDash, prose at
# 4 at once: 103.4 -> 108.8 tok/s, time to first token 590 -> 340 ms; structured at 3 / 4 at once: 175.2 -> 196.3 and
# 196.3 -> 227.9 tok/s; one request unchanged. Exact. MULTI_PREFILL=0 turns it off.
MULTI_PREFILL="${MULTI_PREFILL:-1}"
export TF_GLM_MULTI_PREFILL="$MULTI_PREFILL"
# L2 prefetch in decode windows (patch 0046, adapted from jayleaton/glm53-tensorfold-spark's patch 0460): a side stream
# brings the weights the next kernels read into L2 during each layer's all-gathers. 1 (default): one request's prose
# 48.36 -> 49.46 tok/s, code 59.54 -> 61.08 (two boots each). Same bits. 0: off.
export TF_GLM_L2PF="${TF_GLM_L2PF:-1}"
# The decode expert kernel's trellis loads (patch 0047, adapted from jayleaton/glm53-tensorfold-spark's patch 0580): nc
# (default) as 16-byte non-coherent loads a k step ahead: prose 48.36 -> 48.78 tok/s, code 59.54 -> 60.42 on their own.
# Together with TF_GLM_L2PF=1 and TF_ROCE_MAX_KB=512: one request's prose 49.68, code 61.49 (+2.7% / +3.3%); 4 at once
# prose 74.8 -> 76.6, code 100.0 -> 102.7 tok/s in all (two boots each). Same bits. 0: TensorFold's 32-bit loads.
export TF_GLM_EXL3_LOADS="${TF_GLM_EXL3_LOADS:-nc}"
# Conversations that share a system prompt reuse its prompt state (patch 0015): a 7.9k-token system prompt's second and
# later chats prefill in 0.13 s instead of 4.24 s. Same replies. SHARED_PREFIX=0 turns it off.
SHARED_PREFIX="${SHARED_PREFIX:-1}"
export TF_GLM_SHARED_PREFIX="$SHARED_PREFIX"
# The shared KV pool beyond the 1,048,576-token window (kept prompt states, several long conversations at once) grows
# into the memory left at start. TensorFold sizes it from MemAvailable at start minus MEMORY_RESERVE_GIB
# (TENSORFOLD_MEMORY_RESERVE_GIB), capped at KV_POOL_GIB (its TF_GLM_CACHE_GIB). The server uses about 10 GiB more than
# its own estimate at its peak (a 1M-token prompt), so the reserve sets the lowest free memory on the head: 14.5 leaves
# about 4.5 GiB there, and the pool comes out at ~2.1-2.9M tokens depending on what is free at start. TensorFold's own
# defaults (a tenth of RAM, ~12.2; 3 GiB: pool 1,411,072 tokens) leave more. Raise the reserve if other work shares
# the Sparks' memory.
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-14.5}"
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
KV_POOL_GIB="${KV_POOL_GIB:-12.5}"
export TF_GLM_CACHE_GIB="$KV_POOL_GIB"

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Where rank 1 reads the checkpoint and DFlash2: copy (default) keeps a copy in the worker's own Hugging Face cache
# (prepare.sh copies ~166 GiB over the link); nfs reads the head's HF_CACHE over NFS instead (no copy, no disk on the
# worker), through a read-only docker volume NFS_VOLUME on the worker that prepare.sh creates (no sudo there). The head
# must export NFS_PATH (default: HF_CACHE) to the worker; NFS_SERVER defaults to the head's address on the link.
WORKER_WEIGHTS="${WORKER_WEIGHTS:-copy}"
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"
NFS_VOLUME="${NFS_VOLUME:-glm53-hf}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-glm53}"   # compiled CUDA kernels, a folder per image's patches hash
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53-tensorfold}"   # this recipe's locks and setup marker
# Free disk prepare.sh asks for before it downloads or copies: the checkpoint (~176 GB) under HF_CACHE (on the worker,
# the copy is checked against its size instead), and an image build or copy (~25 GB) under Docker's root on each Spark;
# both together when they share a filesystem.
MIN_FREE_GB="${MIN_FREE_GB:-180}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { local id=${1:-$MODEL_ID}; echo "$HF_CACHE/hub/models--${id//\//--}"; }
# model_revision <id>: the pinned revision of MODEL_ID or DFLASH2_ID (empty: none, the cache's refs/main counts)
model_revision() { if [[ "$1" == "$MODEL_ID" ]]; then echo "$MODEL_REVISION"; elif [[ "$1" == "$DFLASH2_ID" ]]; then echo "$DFLASH2_REVISION"; fi; }
# snapshot_rev <id>: the snapshot this setup serves: the pin, else what refs/main names on this Spark
snapshot_rev() { local rev; rev=$(model_revision "$1"); [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir "$1")/refs/main" 2>/dev/null); echo "$rev"; }

# What scripts/prepare.sh last left ready on both Sparks (it writes this line to PREPARED_MARKER when it succeeds);
# start.sh runs prepare.sh again whenever the current line differs: a missing or different image on either Spark, new
# patches, another model, drafter or revision, another worker. Needs scripts/nodes.sh (the worker's image).
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label wlabel
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  wlabel=$(worker docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  echo "model=$MODEL_ID@$MODEL_REVISION drafter=$DRAFTER@$DFLASH2_REVISION image=$label worker=$wlabel patches=$hash worker_host=$WORKER weights=$WORKER_WEIGHTS"
}
