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

# The Sparks: this machine serves rank 0 and the API; WORKER (ssh target, key-based) runs rank 1. TP: how many Sparks
# (2, the default; 3 through ./start-tp3.sh, experimental: README "3 Sparks"), with WORKER2 (rank 2); a start uses
# WORKER .. WORKER<TP-1> and leaves later ones out (stop.sh stops every configured one).
TP="${TP:-2}"
WORKER="${WORKER:-}"                 # e.g. user@<worker address>; set it in scripts/local.sh
FABRIC_PEER="${FABRIC_PEER:-}"       # the worker's CX7 address when WORKER is reached over another network
WORKER_HF_CACHE="${WORKER_HF_CACHE:-}"  # the worker's Hugging Face cache when it is not its HF_HOME (absolute path)
WORKER2="${WORKER2:-}"; FABRIC_PEER2="${FABRIC_PEER2:-}"; WORKER_HF_CACHE2="${WORKER_HF_CACHE2:-}"   # rank 2, as WORKER / FABRIC_PEER / WORKER_HF_CACHE
MASTER_PORT="${MASTER_PORT:-29551}"  # TensorFold's rendezvous port between the ranks (keep it on the private link)
# The rendezvous address (rank 0's, --master): at TP=2 the head's address on the link to the worker; at TP>2 this
# node's LAN address (what its hostname resolves to), which every worker reaches. SOCKET_IFNAME (TP>2): the netdev of
# NCCL's bootstrap socket on every node (default: each node's default-route netdev).
_ma=""
if [[ "$TP" != 2 ]]; then
  # (|| true: a hostname that does not resolve fails getent, and under pipefail that ended the script silently)
  _ma=$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}' || true)
  [[ -n "$_ma" ]] || _ma=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
MASTER_ADDR="${MASTER_ADDR:-$_ma}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"

# ABLIT=1 serves the Ablit weights, Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit (the same checkpoint,
# abliterated; README "Ablit weights"), instead of the published checkpoint; 0 (default) the published one. The Ablit
# repository is gated: HF_TOKEN must be set (a Hugging Face access token whose account accepted the terms on the
# model's page), or prepare.sh, start.sh and start-tp3.sh stop and say so. Switching downloads the other checkpoint
# (~176 GB). With ABLIT=1, THINKING defaults to 0 (below).
ABLIT="${ABLIT:-0}"
ABLIT_ID="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit"
_id="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"; [[ "$ABLIT" == 1 ]] && _id=$ABLIT_ID
MODEL_ID="${MODEL_ID:-$_id}"   # EXL3 routed experts (4 bpw), BF16 elsewhere
# The checkpoint's revision (a Hugging Face commit sha; DFLASH2_REVISION below is DFlash2's): the one this recipe was
# measured with. prepare.sh downloads exactly it, start.sh serves that snapshot from the local cache (no network), and
# a new upstream commit changes nothing here until the pin does. Empty: the Hub's main when first downloaded. The pins
# belong to the checkpoints above; another MODEL_ID gets no pin unless you set one.
case "$MODEL_ID" in
  Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) _rev=078455ffe6472f9a52fbc1139f58b9db2881b25c ;;
  Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit) _rev=57edefd2f5d9b371c8345883304d5af68b52fa24 ;;
  *) _rev="" ;;
esac
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
# The same image serves two and three Sparks.
IMAGE_TAG="${IMAGE_TAG:-v0.6.0-a1897d591f70}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:bc34d7d63f978cf601f42863b284bc95a567c50c10e9adb0866a635be568bf5f}"
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
# a round): 1 to 8 (patch 0069), with DRAFTER=dflash2 only (mtp: 1). Default 4 on two Sparks, 8 on three (v1.5).
# sparkDash aggregate decode at 4 / 8 requests at once: two Sparks prose 103.2 / 130.8 tok/s, code 126.7 / 167.0; three
# Sparks prose 121.8 / 166.0, code 165.3 / 211.5. One request alone decodes as fast either way. The memory reserve
# grows with PARALLEL (below), so the shared pool shrinks: two Sparks ~1.5M tokens at 8 (2.0-2.6M at 4), three ~4M.
if [[ "$DRAFTER" != dflash2 ]]; then _par=1; elif [[ "${TP:-2}" == 3 ]]; then _par=8; else _par=4; fi
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
# Think before answering by default (0: answer directly unless a request asks to think); off by default with the Ablit
# weights (ABLIT=1), which give their best results without thinking
_think=1; [[ "$MODEL_ID" == "$ABLIT_ID" ]] && _think=0
THINKING="${THINKING:-$_think}"
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
# for all. Same bits. start-tp3.sh defaults to nccl; with roce there, each peer goes over the devices that share its
# subnet (the TP-N engine's RoCE; TF_ROCE_HCA lists them all, the GID is found per device).
COMM="${COMM:-roce}"
export TF_GLM_COMM="$COMM"
# The largest all-gather in KiB that goes over RoCE (patch 0006 reads it; a setting, no patch of its own): 512 (default;
# TensorFold's is 256) also takes the 17-32-row verify windows of concurrent requests: code at 4 streams +1.4%, prose
# +0.5-1% (two boots each). Same bits.
export TF_ROCE_MAX_KB="${TF_ROCE_MAX_KB:-512}"
# How long a RoCE all-gather waits for the other Spark before it fails, in seconds (patch 0052 reads it; 1 to 3600):
# 300 here (the patch's own default is 20). Issue #54: on long prompts (~475k-500k tokens, or after many hours) a rank
# failed after its 20 s while its own writes had all completed, i.e. the peer was late, not lost, and both Sparks went
# down. A late peer now costs a slow round instead; a rank that is really gone is noticed after 300 s instead of 20,
# like the watchdog's stall report (TF_GLM_MULTI_WATCHDOG_S, 300). NCCL's own gathers have no limit at all.
export TF_ROCE_WAIT_S="${TF_ROCE_WAIT_S:-300}"
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
# --parallel: rows of every request's verify window together in a round (patch 0069; TensorFold's TF_GLM_MULTI_WINDOW):
# 16 to 64 in steps of 8; default 64 past 4 requests, else 32. With more requests at once each one's drafts get fewer of
# these rows (8 at 32: ~3 drafts each): at 8 requests, code 141.2 (32 rows) / 160.3 (48) / 167.0 (64) tok/s, prose the
# same at any size (two Sparks). Past 32 rows a round's all-gathers exceed 512 KiB (16 KiB a row): TF_ROCE_MAX_KB rises
# with it (64 rows: 1024) unless you set it. Exact at any size; 64 rows add ~250 MiB a Spark to the estimate.
if (( PARALLEL > 4 )); then _win=64; else _win=32; fi
export TF_GLM_MULTI_WINDOW="${TF_GLM_MULTI_WINDOW:-$_win}"
if (( TF_GLM_MULTI_WINDOW > 32 )) && [[ "$TF_ROCE_MAX_KB" == 512 ]]; then
  export TF_ROCE_MAX_KB=$(( TF_GLM_MULTI_WINDOW * 16 ))
fi
# Kept prompt states (TensorFold's TF_GLM_CACHE_ENTRIES, 8 by default): the oldest is dropped past this count, however
# much of the pool is free. A drafted agent request keeps 1 to 3 (its own plus shared-prefix states), so 8 let three
# or four alternating conversations push each other out (issue #17). Each entry reserves its fixed state (~45 MiB) at
# start: 32 takes ~1 GiB more than 8.
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-32}"
# TF_GLM_CACHE_SHARE_PCT (patch 0097, off by default; issue #84 by @jdecker76): the count above is also a memory
# reservation (the pool shrinks by each entry's fixed state, ~91 MiB measured on two Sparks at the defaults; the
# estimate is unchanged), so a pool with millions of free tokens
# can still evict a live conversation at the 33rd state. A share makes the count at least that part of KV_POOL_GIB's
# budget (~7.5 GiB on two Sparks at the defaults, where 32 states already take ~38%: 50 is ~42 states, 75 ~63). Any count is cut to 75% of the budget, logged at start. Watch
# /health kept_prompts against pool_free_tokens: kept_prompts pinned at the count with a free pool is this symptom.
export TF_GLM_CACHE_SHARE_PCT="${TF_GLM_CACHE_SHARE_PCT:-0}"
# Two more limits on the kept states (patch 0089, both off by default; PR #65 by Thomas Wade): TF_GLM_KEPT_BYTES_GIB
# caps the device memory the kept states own outside the pool (their DFlash2 window copies and recurrent state, ~45 MiB
# each without a window, hundreds of MB with long ones), together: 0 is no cap, 4 is the author's value; past it the
# least valuable state is dropped (by the same order as the entry count's) and freed blocks go back to the driver.
# TF_GLM_KEEP_PER_CHAT keeps at most this many of one conversation's own turn-boundary states (the author runs 2; 0 is
# no limit): one long chat can no longer fill the entries alone. Neither is written to the spill tier when it drops.
# /health shows kept_bytes, kept_bytes_cap and kept_mix.
export TF_GLM_KEPT_BYTES_GIB="${TF_GLM_KEPT_BYTES_GIB:-0}"
export TF_GLM_KEEP_PER_CHAT="${TF_GLM_KEEP_PER_CHAT:-0}"
# Earlier turns keep their reasoning in the prompt (patch 0060), as in zai-org's current template. 1: drop it, as the
# checkpoint's template does; agents then prefill the previous turn's tool loop again at each new user message.
export TF_GLM_CLEAR_THINKING="${TF_GLM_CLEAR_THINKING:-0}"
# The reasoning-effort line ("<|system|>Reasoning Effort: Max") is token 3 of the checkpoint's prompt, so switching effort
# or thinking on/off changes the whole prompt and misses the kept-prompt cache (issue #93, by jdecker76). 1 (patch 0096):
# render it at the tail instead, just before "<|assistant|><think>", so the conversation stays identical across the
# switches (thinking off has no line, as before). Off by default: it moves a line the model was trained to see first, so
# check the effort levels still answer differently (tools/toolcheck.py, tools/end_of_turn.py) before relying on it.
# Not a request field: it changes every thinking-on prompt, and every rank reads it with the other TF_GLM_* switches.
export TF_GLM_EFFORT_TAIL="${TF_GLM_EFFORT_TAIL:-0}"
# Admission at saturation (patch 0075): past the lanes plus MAX_QUEUED a foreground request is refused with
# 429 + Retry-After (529 overloaded_error through the Anthropic bridge) instead of queueing invisibly. Unset
# queues as every scheduler always has; 0 refuses anything past the lanes. A single-instance deployment with
# no load balancer in front of it should set a small value (ours: 0, in scripts/local.sh).
MAX_QUEUED="${MAX_QUEUED:-}"
# A streamed request is checked before its 200 goes out (patch 0095), so it gets the same 429 + Retry-After.
export TF_GLM_MAX_QUEUED="$MAX_QUEUED"
# Sampled decode with the checkpoint's defaults (temperature 1, top_p 0.95, top_k 0) draws from the top_p nucleus
# (patch 0034). Stock TensorFold tests each rank's candidates alone, which fails on any two ranks whenever both hold
# part of the nucleus, so every such step gathered and sorted the whole vocabulary on the CPU: ~15 tok/s against
# greedy's ~58 on two Sparks (issue #91). NUCLEUS_UNION=1 (the default) tests the ranks' candidates together: the
# same nucleus and the same draw (the merged walk stops above every partial rank's last candidate, so no unsent token
# can be in it; tools/test_nucleus_union.py compares both paths), 42-61 tok/s. A step whose nucleus is still wider
# makes one extra 16,384-candidate gather before the whole shards. 0: stock behaviour. Every rank gets the same value.
export TENSORFOLD_NUCLEUS_UNION="${NUCLEUS_UNION:-${TENSORFOLD_NUCLEUS_UNION:-1}}"
# Waiting prompts filled together in one forward (patch 0049): shared work (expert weights, glue, projections) runs once
# for every waiting prompt, attention per prompt on its own state, so each gets the bits it gets alone. sparkDash, prose at
# 4 at once: 103.4 -> 108.8 tok/s, time to first token 590 -> 340 ms; structured at 3 / 4 at once: 175.2 -> 196.3 and
# 196.3 -> 227.9 tok/s; one request unchanged. Exact. MULTI_PREFILL=0 turns it off.
MULTI_PREFILL="${MULTI_PREFILL:-1}"
export TF_GLM_MULTI_PREFILL="$MULTI_PREFILL"
# Smooth streaming (patch 0061): with drafts a round accepts ~3 tokens at once, so a streamed reply arrives in bursts
# (every ~50 ms alone, ~100 ms with 4 streams), and pauses while another request's prompt fills. 1 (default): tokens
# go out one event each at a steady pace from a playout buffer of STREAM_SMOOTH_MS (text appears that much later; the
# reply still ends when it did); the same text, tool calls in place. 0: one event a round, as soon as it is decoded.
STREAM_SMOOTH="${STREAM_SMOOTH:-1}"
STREAM_SMOOTH_MS="${STREAM_SMOOTH_MS:-400}"
export TF_GLM_STREAM_SMOOTH="$STREAM_SMOOTH" TF_GLM_STREAM_SMOOTH_MS="$STREAM_SMOOTH_MS"
# Concurrent prompt fills in layer slices (patch 0062): while other requests decode, a new prompt's FILL_ROWS chunks
# run a few layers at a time (about FILL_BUDGET_MS each) with a decode round between slices, instead of freezing the
# other replies for a whole chunk. FILL_DRAFTS=1 (default): those rounds draft as usual (~3 tokens a stream);
# 0: one token a stream (the new prompt's first token sooner, the others slower while it fills). FILL_BUDGET_MS=0:
# whole chunks, as before. A request alone fills at full speed either way. Same replies.
FILL_BUDGET_MS="${FILL_BUDGET_MS:-200}"
FILL_DRAFTS="${FILL_DRAFTS:-1}"
# FILL_ROWS (TensorFold's TF_GLM_FILL_ROWS, whose own default is 1,024): the rows of such a fill's chunk. 2,048
# (default) is the whole prompt chunk (TF_GLM_PREFILL_ROWS, 2,048), whose buffers already exist: the same memory and
# the same replies (a token's result does not depend on the chunk it is filled in), half the passes over the weights.
# Three Sparks, PARALLEL=4, a prompt filling while another reply streams: a warm +20k turn on a 596k conversation
# 15.6-15.7 -> 14.0-14.1 s, a cold 256k prompt 132.2-132.6 -> 113.4-114.0 s; the streaming reply's gaps the same
# (max 25-67 ms either way). At most the prompt chunk (the engine refuses more): a smaller TF_GLM_PREFILL_ROWS lowers
# the default with it. 1024: as before.
_fill_max="${TF_GLM_PREFILL_ROWS:-2048}"
FILL_ROWS="${FILL_ROWS:-${TF_GLM_FILL_ROWS:-$(( _fill_max < 2048 ? _fill_max : 2048 ))}}"
export TF_GLM_FILL_BUDGET_MS="$FILL_BUDGET_MS" TF_GLM_FILL_DRAFTS="$FILL_DRAFTS" TF_GLM_FILL_ROWS="$FILL_ROWS"
# L2 prefetch in decode windows (patch 0046, adapted from jayleaton/glm53-tensorfold-spark's patch 0460): a side stream
# brings the weights the next kernels read into L2 during each layer's all-gathers. 1 (default): one request's prose
# 48.36 -> 49.46 tok/s, code 59.54 -> 61.08 (two boots each). Same bits. 0: off.
export TF_GLM_L2PF="${TF_GLM_L2PF:-1}"
# The decode expert kernel's trellis loads (patch 0047, adapted from jayleaton/glm53-tensorfold-spark's patch 0580): nc
# (default) as 16-byte non-coherent loads a k step ahead: prose 48.36 -> 48.78 tok/s, code 59.54 -> 60.42 on their own.
# Together with TF_GLM_L2PF=1 and TF_ROCE_MAX_KB=512: one request's prose 49.68, code 61.49 (+2.7% / +3.3%); 4 at once
# prose 74.8 -> 76.6, code 100.0 -> 102.7 tok/s in all (two boots each). Same bits. 0: TensorFold's 32-bit loads.
export TF_GLM_EXL3_LOADS="${TF_GLM_EXL3_LOADS:-nc}"
# The decode expert kernel's block launch order (patch 0090, by lukaszraczylo): 0 (default) the grid as launched, items
# fastest; 1: the eight 128-column blocks of one k slice run together (contiguous trellis reads), then the items; 2: then
# the matrices and splits. The same bits for every value; the author's single-stream gain for 1 is +2.2%, one tester, not yet
# measured here. Not TF_GLM_EXL3_ORDER, which is patch 0020's prompt order (default on).
export TF_GLM_EXL3_DEC_ORDER="${TF_GLM_EXL3_DEC_ORDER:-0}"
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
# Past 4 requests, or a verify window past 32 rows, the server takes more than its estimate counts (measured on two
# Sparks, rank 0's lowest free memory: PARALLEL=4 10.8 GiB, PARALLEL=8 7.0, PARALLEL=8 with a 64-row window 5.8): the
# reserve grows by that much (~0.95 GiB a request past 4, ~0.04 GiB a row past 32), so the lowest free memory stays
# where PARALLEL=4 has it; the pool shrinks instead.
_extra=$(awk -v p="$PARALLEL" -v w="${TF_GLM_MULTI_WINDOW:-32}" 'BEGIN { e = 0; if (p > 4) e += 0.95 * (p - 4);
  if (w > 32) e += 0.04 * (w - 32); printf "%.1f", e }')
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-$(awk -v e="$_extra" 'BEGIN { printf "%.1f", 14.5 + e }')}"
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
# With more Sparks each holds fewer weights, so the pool can take more (the per-token KV cost is the same on every
# rank: the latent cache is replicated). TP=3: 32 GiB leaves rank 0, the busiest, ~5 GiB under a 1M-token prompt
# (at 27: 10.6 GiB lowest on rank 0, pool 5,257,216 tokens; at 32: 5,959,680).
case "$TP" in 3) _pool=32 ;; *) _pool=12.5 ;; esac
KV_POOL_GIB="${KV_POOL_GIB:-$_pool}"
export TF_GLM_CACHE_GIB="$KV_POOL_GIB"
# The display reservation in the pool (patch 0072, PARALLEL above 1): the GB10 firmware keeps ~2 GiB for a screen that
# a headless Spark never uses and MemAvailable never counts. DISPLAY_KV_MIB of it (a multiple of 16, at most 2032;
# 2048 failed ENOMEM in the vLLM kit's #234; 1792 measured here) joins the shared pool on every rank, on top of
# KV_POOL_GIB, without taking host memory: 1792 adds ~277k tokens at PARALLEL=8 (276,480-278,528 with the pool's size).
# Same replies, decode and prefill. Headless Sparks only: start.sh and each rank refuse it while a display is connected
# to card0. Needs /dev/dri/card0 in the containers (nvidia_drm with modeset=1; --gpus all passes it). 0 (default): off.
DISPLAY_KV_MIB="${DISPLAY_KV_MIB:-0}"
export TF_GLM_DISPLAY_KV_MIB="$DISPLAY_KV_MIB"
# Where the span's reservation half comes from (patch 0087): drm (default) maps a DRM dumb buffer on card0, as above;
# dispram takes it from the dispramd daemon of kindling spark-os, whose nvidia_drm runs without modeset (no dumb
# buffers: CREATE_DUMB fails ENOSYS) and whose daemon owns the reservation. start.sh mounts dispramd's socket and its
# python client into every rank for dispram and refuses a Spark without them.
DISPLAY_KV_BACKEND="${DISPLAY_KV_BACKEND:-drm}"
export TF_GLM_DISPLAY_KV_BACKEND="$DISPLAY_KV_BACKEND"

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Where rank 1 reads the checkpoint and DFlash2: copy (default) keeps a copy in the worker's own Hugging Face cache
# (prepare.sh copies ~166 GiB over the link); nfs reads the head's HF_CACHE over NFS instead (no copy, no disk on the
# worker), through a read-only docker volume NFS_VOLUME on the worker that prepare.sh creates (no sudo there). The head
# must export NFS_PATH (default: HF_CACHE) to the worker; NFS_SERVER defaults to the head's address on the link.
WORKER_WEIGHTS="${WORKER_WEIGHTS:-copy}"
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"
# The third Spark (TP=3): WORKER_WEIGHTS2 (default: WORKER_WEIGHTS) and NFS_SERVER2 (default: the head's address on
# that worker's link); NFS_PATH and NFS_VOLUME are the same for all.
WORKER_WEIGHTS2="${WORKER_WEIGHTS2:-}"; NFS_SERVER2="${NFS_SERVER2:-}"
NFS_VOLUME="${NFS_VOLUME:-glm53-hf}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-glm53}"   # compiled CUDA kernels, a folder per image's patches hash
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53-tensorfold}"   # this recipe's locks and setup marker
# Server logs: stop.sh (and start.sh, before it removes a stopped container left from an earlier run) saves each rank's
# container log, stdout and stderr with timestamps, gzipped, as <date>-<time>-rank<N>.log.gz in LOG_DIR here and in
# ~/.cache/tensorfold-glm53/logs on each worker, and keeps the newest LOG_KEEP (0: saves none). docker rm deletes a
# container's own log, so without this a crash's log is gone at the next stop or start.
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-glm53/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
# Free disk prepare.sh asks for before it downloads or copies: under HF_CACHE, what the download still needs (the
# revisions' files whose blobs are not cached yet, from the Hub's file list, plus 5 GB), or MIN_FREE_GB for the
# checkpoint (~176 GB) when that list cannot be read (no huggingface_hub on the host, or no network); on each worker,
# what rsync must send plus 5 GB; and an image build or copy (~25 GB) under Docker's root on each Spark; both together
# when they share a filesystem.
MIN_FREE_GB="${MIN_FREE_GB:-180}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

# HF_TOKEN from scripts/local.sh reaches the hf CLI and the download container only when exported
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
# need_hf_token: stop before anything else when the checkpoint is the gated Ablit one (ABLIT=1) and HF_TOKEN is not set
need_hf_token() {
  [[ "$ABLIT" =~ ^[01]$ ]] || die "ABLIT is 0 or 1, not $ABLIT"
  [[ "$MODEL_ID" == "$ABLIT_ID" && -z "${HF_TOKEN:-}" ]] || return 0
  die "The Ablit weights ($ABLIT_ID, ABLIT=1) are gated on Hugging Face.
    1. Open https://huggingface.co/$ABLIT_ID, log in and agree to its terms.
    2. Create a token with read access at https://huggingface.co/settings/tokens (the same account).
    3. Set HF_TOKEN=hf_... in scripts/local.sh, in .env or in the environment, and run this again.
    Or set ABLIT=0 to serve the published checkpoint."
}
model_cache_dir() { local id=${1:-$MODEL_ID}; echo "$HF_CACHE/hub/models--${id//\//--}"; }
# model_revision <id>: the pinned revision of MODEL_ID or DFLASH2_ID (empty: none, the cache's refs/main counts)
model_revision() { if [[ "$1" == "$MODEL_ID" ]]; then echo "$MODEL_REVISION"; elif [[ "$1" == "$DFLASH2_ID" ]]; then echo "$DFLASH2_REVISION"; fi; }
# snapshot_rev <id>: the snapshot this setup serves: the pin, else what refs/main names on this Spark
snapshot_rev() { local rev; rev=$(model_revision "$1"); [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir "$1")/refs/main" 2>/dev/null); echo "$rev"; }

# What scripts/prepare.sh last left ready on every Spark (it writes this line to PREPARED_MARKER when it succeeds);
# start.sh runs prepare.sh again whenever the current line differs: a missing or different image on any Spark, new
# patches, another model, drafter or revision, other workers. Needs scripts/nodes.sh (the workers' images). At TP=2 the
# line is the one it always was.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label wlabel i line
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  wlabel=$(worker 1 docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  line="model=$MODEL_ID@$MODEL_REVISION drafter=$DRAFTER@$DFLASH2_REVISION image=$label worker=$wlabel patches=$hash worker_host=$WORKER weights=$WORKER_WEIGHTS"
  for (( i = 2; i < TP; i++ )); do
    wlabel=$(worker "$i" docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
    line+=" worker$i=$wlabel worker${i}_host=$(worker_host "$i") weights$i=$(worker_weights "$i") nfs$i=$(wval NFS_SERVER "$i")"
  done
  echo "$line"
}

# Spill tier (patch 0088-glm-spill-tier, off by default): kept prompt states go to local disk on each Spark and come
# back instead of a new prefill, also after a clean restart. SPILL_GIB: the cap per Spark (0: off). SPILL_DIR: the same
# absolute path on every Spark (mounted at /spill; files owned by your user). SPILL_HIGHWATER: past this fraction of
# the KV pool, the kept prompts eviction would take next are written in the background (1.0: only when evicted).
# A clean stop writes what is kept within SPILL_FLUSH_S seconds; STOP_TIMEOUT gives it the time. README: Spill tier.
SPILL_GIB="${SPILL_GIB:-0}"
SPILL_DIR="${SPILL_DIR:-$HOME/.cache/tensorfold-spill}"
SPILL_HIGHWATER="${SPILL_HIGHWATER:-0.70}"
SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-8192}"
SPILL_MIN_FREE_GIB="${SPILL_MIN_FREE_GIB:-50}"
SPILL_FLUSH_S="${SPILL_FLUSH_S:-60}"
if [[ "$SPILL_GIB" != 0 ]]; then
  STOP_TIMEOUT="${STOP_TIMEOUT:-$(( ${SPILL_FLUSH_S%.*} + 30 ))}"
fi
