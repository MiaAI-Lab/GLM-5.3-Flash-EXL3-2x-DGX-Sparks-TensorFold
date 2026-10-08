# Changelog

Every change to this recipe, newest first. Each release names the image it serves: `scripts/prepare.sh` pulls
`ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` by the digest pinned in `scripts/config.sh`.

## Unreleased

### Added
- **Spill tier** (patch `0084-glm-spill-tier`; `SPILL_GIB`, `SPILL_DIR`, `SPILL_HIGHWATER`; off by default): a kept
  prompt state that leaves the KV pool is written to local disk on each Spark and read back when a later request
  extends it, also after a clean restart, instead of a new prefill. Past `SPILL_HIGHWATER` (0.70) of the pool the
  states eviction would take next are written early, in the background, so an eviction frees its rows at once; a
  restore reads on a background thread while other streams keep decoding. Every read is checked against per-block
  CRC-32s, and the files are private to you. Prompts with images or video are stored under their pictures'
  content (`PARALLEL` above 1). It works beside `DISPLAY_KV_MIB` (rows in the display reservation go to and from disk
  through a kernel, as the pool's own moves there do). Credits in `NOTICE` and `CREDITS.md`.
  Numbers: README, "Spill tier".

## v1.8 (2026-10-06): pictures read once, quoted markers, capacity refusals, and the take-over memory fix

Image: `v0.6.0-31557ed1cef6` (`sha256:cbb4b3c66273e2965dd40a7227e7a5243db333fe250113fb3987462ad4f12588`), 82 patches, for two and three Sparks (v1.7.1's plus `0078`-`0083`). Every change below was also
tested live on two Sparks before release; results are on the PRs.

### Fixed
- **A fresh conversation after a long one no longer runs both ranks out of memory** (`_take_over`, patch
  `0078-glm-take-over-decide-then-copy`, the same change as upstream ashhart/TensorFold#421). When a conversation
  sharing no prefix with the last one arrived after a long conversation, both ranks died with `NV_ERR_NO_MEMORY`; it
  also happened with the 1M window and `KV_POOL_GIB=3`. `_take_over` cloned every kept state and only then dropped the
  ones over the budget, so reserved memory grew to the sum of all their sizes. It now works out which states stay and
  clones only those; the states that stay and the replies are the same by design (200 random cases against the old
  walk). After a 256k, 21-turn conversation, a fresh one on 2 Sparks: `save_rows` 22 -> 1, reserved +14.8 -> +1.4 GiB,
  `take_over` 3.3 -> 0.10 s; before, the driver logged `NV_ERR_NO_MEMORY` 22 times. After: 3 fresh boots, 5 cycles
  of a long task then a fresh one, 48 benchmark attempts, no allocation failure; pass counts match the unpatched build where it had the same trials (15 of 21 graded at low effort on both; 15 of 21 at high effort against 10 of 11 before). These runs used the v1.4 patch set; patches 0071, 0074 and 0077 change the shared pool and the kept cap (`multi.py`), not `_take_over`, so the patch applies to v1.7.1 unchanged.

### Added
- **`TENSORFOLD_GLM_PICTURE_CACHE`** (patch `0079-glm-picture-cache`, by ThomasWadeZ, #63): the vision frontend reads
  a request's pictures once, not on every turn of the chat. 10 pictures in a chat, a later turn's first token
  0.43-0.46 s -> 0.16-0.19 s; prompt tokens and replies identical. Host memory only, LRU-capped
  (`_PICTURE_CACHE_MB`, default 384; `_PICTURE_CANVASES`, default 8); `0` restores the old path.
- **Quoted media markers** (patch `0080-vision-quoted-markers`, by ThomasWadeZ, #64, needs `0079` first): every
  picture and clip of a request carries a per-request nonce, so text that merely quotes the template's picture span
  stays text. A request quoting it beside a real picture was a 400, now answers; a template or client that drops a
  picture is still refused.
- **Capacity refusals render once** (patch `0081-tfcap-capacity-status`, by johnwhited, #48): 429 with
  `Retry-After: 5` everywhere (was a bare 503 at two sites, a 400 on the generate path).
- **`TF_GLM_MAX_QUEUED`** (patch `0082-tfcap-admission-cap`, by johnwhited, #48): unset queues as every scheduler
  always has; a number refuses foreground requests past the lanes plus that many. With `0`, the fifth request on four
  lanes gets 429 + `Retry-After: 5` at once; unset, it queues as before (38 s in the release check).
- **Delivery abort** (patch `0083-tfcap-delivery-abort`, by johnwhited, #48): a stream whose delivery callback raises
  (broken pipe, reset, timeout) ends at once and frees its lane, instead of the lane leaking until generation ends.

## v1.7.1 (2026-10-05): a new agent run resumes its system prompt again once the kept cap is full

Image: `v0.6.0-1692d2df78d2` (`sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c`), 76 patches, for two and three Sparks (v1.7's image plus `0077`).

### Fixed
- **#75, a new run of an agent no longer resumed its system prompt once the kept cap was full** (patch
  `0077-glm-kept-cap-shared-by-recency`; diagnosed, and the fix proposed, by @meleesciony). Since `0071` every run and
  every cold start keeps its states in an extent of its own, and `0063` never dropped an extent's latest state while
  anything superseded was kept anywhere. Within hours of a start every entry was some conversation's latest state; a
  new run's system-block state was then the only superseded entry and went at its next kept state, so the next run of
  the same agent read its whole prompt again (first token 14 s -> 40 s on 42-49k-token prompts; 15 of 16 runs cold).
  Shared-prefix states now go by recency only, beside the other conversations' latest states; a conversation's earlier
  states still go first. A larger `TF_GLM_CACHE_ENTRIES` now helps too (before, it only delayed this). Same replies.

## v1.7 (2026-10-05): the Ablit weights (`ABLIT=1`, gated: needs `HF_TOKEN`)

Image: `v0.6.0-c4cab25d2d36` (`sha256:b47c19d66633f27cbe37da13fbc580363f466c08b9529feab1eecb1a4b904bf1`), 75 patches, for two and three Sparks (unchanged from v1.6).

### Added
- **`ABLIT=1` serves the Ablit weights**,
  [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit)
  (pinned at `57edefd2`), instead of the published checkpoint; set it in `scripts/local.sh` (now in `local.sh.example`) or `.env`.
  The repository is gated: `start.sh`, `start-tp3.sh` and `prepare.sh` stop when `HF_TOKEN` is not set and say how to
  get one and accept the terms on the model's page, and `prepare.sh` checks that the token reaches the gated files
  before the image and the download (README: Ablit weights), without putting the token on a command line. `HF_TOKEN`
  set in `scripts/local.sh` is now exported to the download.
- **With `ABLIT=1`, thinking is off by default** (`THINKING` defaults to `0`): the Ablit weights give their best
  results answering directly. A request can still ask to think, and `THINKING=1` turns it back on by default.

## v1.6 (2026-10-05): agent sessions keep their history (beside sub-agents and under a full pool), queued requests whose client left are dropped, no raw `<|assistant|>` in replies, the display reservation in the pool, a longer RoCE wait, SPLIT retried at start

Image: `v0.6.0-c4cab25d2d36` (`sha256:b47c19d66633f27cbe37da13fbc580363f466c08b9529feab1eecb1a4b904bf1`), 75 patches, for two and three Sparks.

### Fixed
- **#43: a conversation lost its kept prompt whenever another conversation with the same system prompt resumed from
  it** (patch `0071-glm-shared-prefix-copy`, by @ezoushen, #44). The resume took over the extent that held the shared
  state and evicted the longer states in it, which belong to the conversation that wrote it, so a coding agent re-read
  its whole history after each sub-agent request. The shared rows are now copied into free rows of their own; with no
  free rows, or `TF_GLM_MULTI_LONE=1`, it behaves as before. Placement only: the same replies. Two Sparks, v1.5 +
  0071, `tools/prompt_reuse.py` (new): 5% -> 99% of a ~33k-token turn resumed, 16.5 s -> 1.0 s to the first token
  (measured by @ezoushen and @plotarmordev); the same replies, drafted == serial.
- **#61: two long conversations taking turns at a nearly full pool evicted each other's kept prompt** (patch
  `0074-glm-compact-before-evict`, by @ezoushen, #62). A turn whose rows the pool had free, but not in one range,
  evicted kept prompts until a range opened, and compacted only after evicting them all, so two coding agents at once
  re-read 150-210K tokens of history (2-3 minutes). The pool now moves caches together first (each at most once) and
  evicts only while its free rows fall short. Placement only: the same replies. `tools/pool_pressure.py`: the other
  conversation's next turn 0% -> 100% resumed (measured by @ezoushen, two Sparks); `tools/pool_room_check.py` checks
  the moves on a CPU arena.
- **Requests waiting while every slot was busy kept waiting after their client left** (patch
  `0073-glm-queued-cancellation`, by @desy0305, #51), until a slot freed; the scheduler now drops them at once, in
  queue order. A reply whose connection fails mid-stream ends after the round instead of decoding on (the
  delivery-failure handling from @johnwhited's #48). Measured by @desy0305 on two Sparks: a queued fifth / ninth
  request cancelled with 4 / 8 slots busy was acknowledged in 0.11-0.21 s, the busy replies equal their serial ones,
  `PARALLEL=1` unchanged; `tools/test_queued_cancellation.py` checks it on the CPU.
- **#60: a raw `<|assistant|>` token reached replies** (patch `0075-glm-assistant-ends`; reported by
  @Lukas-tek-no-logic). At high reasoning effort the model sometimes wrote it inside its answer and began a second
  one; it now ends the reply, like the checkpoint's end tokens. `TF_GLM_ASSISTANT_ENDS=0` restores the old behaviour.
  Replies without the token are unchanged.
- **#54: a RoCE all-gather failed on long prompts and took both Sparks down.** `TF_ROCE_WAIT_S` is now 300 s (was the
  patch's 20): in the reports the failing rank's own writes had all completed, so the peer was late rather than lost.
  A late peer now costs a slow round; a rank that is really gone is noticed after 300 s, like the watchdog's report.
- **#36: with `SPLIT=1` a rank's first NCCL connection failed about half the time on some pairs** (NCCL error 2,
  `ibv_reg_mr`: cannot allocate memory, before any weights load). `start.sh` now tries such a start once more as it
  was, then starts with `SPLIT=0` (the same replies, long prompts fill slower) and says so.

### Added
- **`DISPLAY_KV_MIB`** (patch `0072-glm-display-kv`, by @ezoushen, #56; off by default): with `PARALLEL` above 1, up to
  2032 MiB of the GB10's display reservation, which a headless Spark never uses and `MemAvailable` never counts, joins
  the shared pool on every rank without taking host memory. Measured by @ezoushen at `PARALLEL=8`: 1792 MiB adds
  276,480 tokens a boot (1,611,776 -> 1,992,704), the same reply hashes on and off, decode, prefill and the 195k
  needle within boot-to-boot noise. Headless Sparks only: refused while a display is connected; fails closed.
  `tools/display_kv_check.py` runs its checks in the image.

### Checked
The published image on three Sparks (`./start-tp3.sh`, `PARALLEL=8`) and two (`./start.sh`), 2026-10-05, against v1.5:
- Exact: 8 requests at once equal the same requests one at a time (11/11 staggered, 11/11 in a burst, three Sparks);
  the 22 saved serial references byte-identical (two Sparks); drafted == serial 6/6 on three Sparks, two Sparks, the
  small-pool start and `DENSE=fp8`; prefill hashes at 12k / 50k / 149k tokens as v1.5's; the 195k needle (both).
- Speed as v1.5 (sparkDash, three Sparks, 1 / 4 / 8 at once): prose 65.7 / 121.8 / 162.8 tok/s (v1.5 65.7 / 121.8 /
  166.0), code 100.0 / 164.3 / 210.1 (100.4 / 165.3 / 211.5); prefill unchanged.
- #43: `tools/prompt_reuse.py` 5% -> 99% of each turn resumed (16.5 s -> 0.7-1.1 s), three and two Sparks.
- #61: `tools/pool_pressure.py` (`CONTEXT=131072 KV_POOL_GIB=0.5`, `PARALLEL=8`): the other conversation's next turn
  0% resumed (15.8 s) without 0074, 100% (0.6 s) with it.
- #60: its request with `DENSE=fp8`: v1.5's reply carried `<|assistant|>` at character 1438 of 3,849; v1.6's ends
  there, 1,438 characters, no marker, the same reasoning (534 tokens).
- CPU tests as v1.5's plus the new ones (0071's resume, 0075's end tokens), 0073's and 0074's checks in the image; the
  engine GPU tests have v1.5's 14 known failures and no new ones.
- `DISPLAY_KV_MIB` was not measured here: these Sparks run `nvidia_drm` with `modeset=0` (an `/etc/modprobe.d`
  override), and `start.sh` refuses it with that reason. @ezoushen's numbers are from modeset=1 Sparks.
- The `start.sh` retry for #36 was checked offline (a harness with stubbed ranks), not against a real NCCL failure.

## v1.5 (2026-10-03): up to 8 requests at once (8 by default on three Sparks), serial requests stop when their client leaves

Image: `v0.6.0-9f73cca659a1` (`sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237`), 70 patches, for two and three Sparks.

### Added
- **Up to 8 concurrent requests** (patch `0069-glm-eight-streams`): `PARALLEL` takes 1 to 8 (was 1 to 4; above 1 still
  needs `DRAFTER=dflash2`). Default: 4 on two Sparks (unchanged), 8 on three. The batched verify window's segment tables
  and the segmented kernels' launch grids hold one segment a stream past four (four, as before, up to four), and the
  multi-stream DFlash2 drafter, scheduler and startup estimate take 5 to 8 streams. sparkDash aggregate decode, 4 -> 8
  requests at once: two Sparks prose 103.2 -> 130.8 tok/s (+27%), code 126.7 -> 167.0 (+32%); three Sparks prose 121.8
  -> 166.0 (+36%), code 165.3 -> 211.5 (+28%); one request alone unchanged. Replies byte-identical: 22/22 concurrent
  cases equal their serial references with 8 in flight (two and three Sparks, windows 32 and 64), drafted == serial,
  the long-prompt hashes and the 195k needle unchanged; `PARALLEL=4` gives v1.4's shas.
- **`TF_GLM_MULTI_WINDOW`** (patch `0069`): the rows of every request's verify window together in a round, 16 to 64 in
  steps of 8; 32 as before, 64 by default past 4 requests (8 requests' code: 141.2 / 160.3 / 167.0 tok/s at 32 / 48 / 64
  rows; prose the same at any size). `TF_ROCE_MAX_KB` follows it past 32 rows (64: 1024) unless set, keeping a round's
  all-gathers on RoCE.
- **The memory reserve grows with the requests at once** (`MEMORY_RESERVE_GIB`: 14.5, plus ~0.95 GiB a request past 4
  and ~0.04 GiB a window row past 32: 19.6 at 8 requests and 64 rows): more requests take more than the startup
  estimate counts (two Sparks, head's lowest free memory at the same reserve: 10.8 GiB at 4, 7.0 at 8, 5.8 at 8 with 64
  rows). With the grown reserve: two Sparks 12.9 GiB at the lowest under a 195k-token prompt (pool ~1.5M tokens), three
  Sparks 11.5 GiB (pool ~4.0M).

### Fixed
- **#38: at `PARALLEL=1` a request whose client left kept decoding to `MAX_TOKENS`** (patch `0070-glm-serial-stop`), and
  every later request waited behind it; stop strings and gate cuts waited the same way. The serial decode loops did
  not read the request's stop: rank 0's decision now rides on the round's sample all-gather, so both ranks end after
  the same round. Same replies. `--parallel` above 1 without the DFlash2 drafter is refused at start with the options
  (DFlash2, `PARALLEL=1`, or no drafts) instead of a confusing message after loading.

### Changed
- The README no longer offers the earlier TR3-4bpw checkpoint, and `scripts/config.sh` no longer pins its revision:
  the recipe serves `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`. (Any `MODEL_ID` still works, without a pin.)

## v1.4 (2026-10-03): image prompts resume, tool calls never dropped, pictures in tool results, earlier reasoning kept, smooth concurrent streaming, 3 Sparks (experimental)

Image: `v0.6.0-5e01f1bb74d8` (`sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529`), 68 patches, for two and three Sparks.

### Changed
- **Earlier turns' reasoning stays in the prompt** (#23, patch `0060-glm-keep-thinking`, by @kky42), as in zai-org's
  current template. The checkpoint's template dropped it at each new user message, so agents prefilled the previous
  turn's tool loop again (a replayed agent session: 85,127 -> 16,106 tokens). `TF_GLM_CLEAR_THINKING=1`, or a
  request's `chat_template_kwargs.clear_thinking: true`, restores the old rendering. Prompts without earlier-turn
  reasoning render as before.
- **`chat_template_kwargs.thinking`** (#25, patch `0057-server-thinking-alias`, from @Alexbob0's PR) is read as
  `enable_thinking` when that is absent: `true` / `false`, or `{"type": "enabled" | "disabled"}`. DeepSeek-V4 clients
  such as pi send it; it was ignored. Other values (`{"type": "adaptive"}`, a string) leave the server's default, as
  before, and are logged once; they are never refused.

### Added
- **Smooth streaming with several requests at once** (patches `0062-glm-sliced-fill` and `0061-server-smooth-stream`;
  `FILL_BUDGET_MS=200`, `FILL_DRAFTS=1`, `STREAM_SMOOTH=1`, all on by default). Concurrent replies arrived in bursts and
  froze while another request's prompt filled: each 1,024-row prompt chunk (~0.65 s) stopped every other reply.
  Now a new prompt's chunks run a few layers at a time (about `FILL_BUDGET_MS` each) with a drafted decode round between
  slices, and a streamed reply's tokens go out one event each at a steady pace from a 400 ms playout buffer
  (`STREAM_SMOOTH_MS`). Three replies streaming while a fresh ~25k-token prompt arrives (two boots each): the replies'
  gaps p50 / p90 / max **630 / 673-690 / 754-818 ms -> 81 / 149-150 / 230-231 ms**, none above 250 ms (75-78
  before); they received **81 -> 650-662 tokens** during the fill; the new prompt's first token **16.1 -> 21.1-21.3
  s** (`FILL_DRAFTS=0`: 18.1 s with gaps up to ~250 ms; `FILL_BUDGET_MS=0`: the old behaviour). Without a fill, four
  replies at once: gaps p50 / p90 **107 / 128 -> 40 / 58 ms**, one reply **50 / 68 -> 16 / 20 ms**; replies end when
  they did. Replies are byte-identical (22/22 concurrent cases against serial references, drafted == serial, long
  prompts, the needle); a request alone fills as before. Text appears ~0.4 s later than it is decoded; tool calls keep
  their place in the stream.
- **3 Sparks (experimental):** `./start-tp3.sh` (`TP=3`, `COMM=nccl` by default, `roce` allowed), with `WORKER2` for
  rank 2. Links are found per pair of nodes (a triangle of direct cables); `stop.sh` stops every configured worker;
  `prepare.sh` prepares every worker. Each rank's container gets `TF_ROCE_HCA`, the RoCE devices its NCCL uses toward
  its peers (at two Sparks it is not set: the RoCE all-gathers take the same devices from `NCCL_IB_HCA`). The
  PCIe-twin rail by name (#30) applies at two Sparks only: past two, a node's devices are the ones that share a subnet
  with a peer, so a twin without an address in such a subnet is not used. Three Sparks ran on v1.3.2's patches and, on
  2026-10-03, on top of v1.4: exact (concurrent == one at a time 22/22, drafted == serial, sliced fill on == off, the
  195k needle, cancellation mid-fill), measurements in the README. Two Sparks run as before: the same docker commands
  and defaults.
- **The TP-N GLM engine as patches 0066-0068, after 0001-0065:** `0066-glm-tp-n` (the engine on 3 ranks, `--tp 3`:
  heads, expert columns, vocab and DFlash2 KV groups in whole units, remainder to the lowest ranks; per-subnet RoCE,
  b12x's proxy modified for more than two Sparks), `0067-glm-tp3-split-pad` (at `TP=3` a 2,048-row prompt chunk pads
  to 2,049 rows, which the buffers did not hold, so the split prefill never ran: one pad row),
  `0068-glm-tpn-split-buffer-rows` (the memory estimate counts that row). On v1.4's code: the early split connection
  (0064, #36) opens one to every peer past two ranks instead of being skipped there, and the rank checks (0065, #29)
  name the follower's own rank in their errors. Two Sparks: buffers, estimate and connections unchanged.
- `DRY_RUN=1 ./start.sh` prints every rank's docker command and exits without stopping or starting anything;
  `DRY_RUN=1 ./stop.sh` says what it would stop on which Spark and stops nothing.
- **Per-worker settings** for the third Spark, as v1.3.3 / v1.4 do for the one worker: `FABRIC_PEER2`,
  `WORKER_HF_CACHE2`, `WORKER_WEIGHTS2`, `NFS_SERVER2`; every rank's log saved before its container is removed
  (`stop.sh`, and `start.sh` for a stopped container left from an earlier run); the image checked by content on every
  worker (#8); every worker's file list compared in byte order (#21) with only what `rsync` must send counted (#24); a
  `FABRIC_PEER2` host name resolved as at two Sparks.
- `KV_POOL_GIB` defaults per `TP`: 12.5 at two Sparks (unchanged), 32 at three (a 1M needle at 27 left rank 0 10.6 GiB
  at its lowest, so ~5.6 at 32).

### Fixed
- **#36, `SPLIT=1` failed to start on some Spark pairs** (NCCL `ibv_reg_mr ... Cannot allocate memory` on rank 1):
  the split's send/receive connection was opened after the cache pool had taken the memory. It now opens before the
  weights load (patch `0064-glm-split-connect-early`). Same replies.
- **#29, a stalled server said nothing** (patch `0065-glm-rank-checks`). Not the fix: the cause is not found yet. Each
  step's message between the ranks now carries a sequence number and a checksum, so ranks that fall out of step stop
  with an error naming it; `/health` reports `iteration_s`, how long the current step has run; and
  `TF_GLM_MULTI_WATCHDOG_EXIT=1` makes a server stuck past `TF_GLM_MULTI_WATCHDOG_S` (300 s) exit after printing every
  thread's stack, so a supervisor can restart it. Same replies.
- **An agent's tool loop pushed other conversations' kept prompts out** (PR #32 by @Alexbob0, patch
  `0063-glm-kept-cap-superseded-first`): past `TF_GLM_CACHE_ENTRIES` the least recently used state went, so a
  conversation taking many short turns dropped the only state of another one waiting on a long reply, whose next turn
  then read its whole history again. A conversation's earlier states, superseded by its own longer ones, now go first.
  Same replies.
- **Downloads without the host `hf` CLI left the Hugging Face cache root-owned** (PR #34 by @100menotu001): the
  fallback container ran as root, so later `hf` runs, the copy to the worker and cache cleanup failed with Permission
  denied. It now runs as your user, with the same `hub/` layout.
- **#11, a prompt with a picture anywhere in it never resumed from a kept state** under `PARALLEL` above 1: every turn
  of a conversation with pictures in its history read the whole history again (minutes at 100k+ tokens). Kept states
  are now told apart by each picture's content, and only the pictures past the resume point are encoded again
  (patch `0054-glm-image-prompt-reuse`, by @abhicnv007, applied as contributed with two review changes). Also
  reported, with measurements, by @DevRico003, @lukemdanastasi and @d4rkdpg.
- **#22, a tool call the model ended with its end token before `</tool_call>` was dropped** (`finish_reason: "stop"`,
  no call, no text). It is now closed and sent as a call when it then parses, else returned as text; a call whose
  `<arg_key>` the model wrote as whitespace gets it back (patch `0055-glm-open-tool-calls`). A call cut by the token
  limit is still never sent. Reported by @teamlewis-bot, confirmed by @andrejsstepanovs.
- **#28, pictures in tool results were refused with HTTP 400** (`image_url parts are supported only in user
  messages`), which every later request of an agent session replayed. They are now read in place, inside their
  `<tool_response>`, as the served template renders a tool result's media (patch `0056-glm-tool-result-media`).
  With `VISION=0` they become the template's own "unable to process this image" reminder, so the session goes on; a
  picture in a user message still needs `VISION=1` (HTTP 400 that says so). Reported by @d4rkdpg.
- A POST to an unknown route left its body on the kept-alive connection, so the next request on it failed with 400
  (TensorFold v0.6.1's fix for #181, commit 50dfe38a, backported: patch `0059-server-refused-bodies`).
- A client that left was not noticed once the server held more than ~1,000 descriptors, and its reply was decoded for
  nobody (TensorFold PR #218 by @jayleaton: patch `0058-server-client-gone-poll`).

## v1.3.3 (2026-10-02): our own checkpoint by default, setup fixes, saved server logs, both PCIe links of a QSFP port

Image unchanged: `v0.6.0-ae8d1c789b47`. No patch changes. The default checkpoint changes, so replies differ from v1.3.2
(with `MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` they are the same as before).

### Changed
- **Default checkpoint: [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)** (rev `078455ff`), Mia's AI Lab's own EXL3
  quantization of GLM-5.3-Flash, Apache-2.0. Same format, size (~176 GB), speed and memory as TR3-4bpw. Against TR3-4bpw on
  the same build: KL divergence to Z.AI's official FP8 release 2-18% lower as served, on all six test sets (paired 95%
  intervals exclude zero on all but chat); coding equal (HumanEval+ and MBPP+, thinking on: 469 vs 468 of 542, paired p = 1.0) with
  ~10% shorter replies; GSM8K 247 vs 245 of 250, HumanEval 157 vs 160 of 164 (neither significant). Details on its
  model card. TR3-4bpw stays pinned and one setting away: `MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`.
- **Disk:** the first `./start.sh` after updating downloads the new checkpoint (~176 GB) and copies it to the worker.
  To free TR3's space afterwards, delete `models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw` from the Hugging Face cache
  on both Sparks (`~/.cache/huggingface/hub` unless `HF_HOME` / `WORKER_HF_CACHE` say otherwise); the recipe never
  deletes it itself.

### Added
- **Saved server logs.** `docker rm -f` deletes a container's log, so a crash's log was lost at the next stop or
  restart. `stop.sh`, and `start.sh` before it removes a stopped container left from an earlier run, now save each
  rank's log (stdout and stderr, with timestamps), gzipped, as `<date>-<time>-rank<N>.log.gz` in
  `~/.cache/tensorfold-glm53/logs` on each Spark (`LOG_DIR` on the head), keep the newest 10 of each rank
  (`LOG_KEEP`; `0` saves none), and print where. A log that cannot be saved only warns.
- **`WORKER` and `FABRIC_PEER` as host names** (#26): a name (an `/etc/hosts` alias of the CX7 address, a LAN name,
  mDNS) is resolved to IPv4 before the route lookup, which takes addresses only; before, it stopped with `no route
  from this node to <name>`. By @Alexbob0.
- **`WORKER_HF_CACHE`** (#26): the worker's Hugging Face cache when it is not its `HF_HOME` (a shared models folder,
  say), used by the copy, the checks and rank 1's mount. Unset, nothing changes. By @Alexbob0.
- **`tools/end_of_turn.py`** (#27): the end-of-turn check behind #18 (8 short French coding prompts, thinking off:
  replies cut by `max_tokens`, and P(end of turn) right after the closing code fence); exit code 1 above `max_cut`
  cut replies. By @Alexbob0.

### Changed (setup)
- **Both PCIe links of the cabled QSFP port** (#30): a Spark's QSFP port reaches the GB10 over two PCIe Gen5 x4
  links, so it shows up as two netdevs and two RoCE devices ("twins"). The rails scan only took a second device in
  the link's own subnet, so with the twins in different subnets (NVIDIA's two-Spark playbook) NCCL and the RoCE
  all-gathers ran on one x4. The twin is now paired by name as well. By @webzone. With a fix of ours: when the twins
  share the link's subnet (as on our pair), the twin was listed twice (`rocep1s0f1,roceP2p1s0f1,roceP2p1s0f1`); it
  is listed once. On our pair the devices are unchanged (head `rocep1s0f1,roceP2p1s0f1`, worker
  `rocep1s0f0,roceP2p1s0f0`, as before). `NCCL_RAILS=1` still uses one device.

### Fixed
- **#21, the worker copy check failing when the two Sparks sort differently.** The check compared the head's and the
  worker's `find | sort` manifests as strings, each sorted in its own host's locale (ssh carries none of ours), so
  identical copies failed after the full copy whenever the locales differed or the launcher set none (systemd, cron,
  Tailscale SSH). Every manifest is now sorted in byte order (`LC_ALL=C`), inside the commands sent to the worker too,
  and a mismatch prints the first differing lines. Reported by @alexandrupetraru, reproduced by @ThinkCode.
- **#24, the disk checks asking for the whole checkpoint when it is already cached.** The head asked for
  `MIN_FREE_GB` (180 GB) whenever the pinned revision's snapshot was new, even when every blob was in the cache from
  an earlier revision. It now asks the Hub which files the revisions hold and counts only those whose blob is missing
  (by name and size), plus 5 GB; that needs `huggingface_hub` on the host (`python3`, or the `hf` CLI's own Python),
  and without it or the Hub the old rule applies. The worker check counted the whole snapshot; it now counts what
  `rsync` must send. Reported by @ThinkCode.

### Unchanged
- The image, the patches and every serving setting besides the checkpoint: memory and speed are those of v1.3.2
  (except on pairs whose port twins were left out before, #30); with `MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
  replies are too.
- `start.sh`'s help said `TF_GLM_MULTI_LONE` defaults to 1; it has been 0 since v1.3.1 (help text only).

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
