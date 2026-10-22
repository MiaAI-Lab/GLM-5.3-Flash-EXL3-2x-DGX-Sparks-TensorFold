# Registered-memory TP4 RDMA: compact historical evidence

Four-host, two-rail-per-neighbor GB10 ring qualification passed for native source `be94e919e35c628261bd38364b1743a2e4353a26`. **Switched hardware remains unverified.** These files summarize historical receipts and make no assertion of current serving state. No tests, builds, live checks or verifier runs were performed for this packaging change.

The tested native Zig profile uses context **131072**, parallel **8**, maximum output **32768**, DFlash2/FP8 KV, CLI prompt cache **3 GiB**, environment cache **32 GiB**, L2PF1/SIDE1 and registered 512 KiB slots. Larger/ineligible gathers use NCCL. Exact checkpoint, recipe/patch, engine/probe/image, AOT, loaded NCCL and toolchain identities are in [qualification.json](qualification.json). The separately patched NCCL library is a prerequisite; the opt-in flag does not install it or qualify the stock wheel.

[benchmark.json](benchmark.json) retains the actual Grill 0.3.0 C1/exact400 workload/request settings, all eight acquisition medians per transport, counts, raw capture/report hashes and estimator definitions. Same-binary sequential captures recorded **157.09 NCCL versus 184.41 registered-RDMA median acquisition tokens/s**. The official log-mean effect was **+16.99%**, nominal model-based 95% interval **[15.79%, 18.21%]**. The displayed medians and effect use different estimators. Rates include TTFT and settlement. Sequential short synthetic captures do not establish causality or general serving speed; drift, carryover and correlated variation can invalidate the nominal interval's precision.

[qualification.json](qualification.json) preserves the executed host/semantic/launcher counts and placeholder limit; four-rank byte/graph/NCCL-oracle results; 64 mixed replays with 128 own operations and 256 relay stripes per registered rank; direct-on-ring refusal; and 40 exact outputs across NCCL, registered graph and eager. Final containers of the same profile separately received 31 settled acceptance cases, 12 exact comparisons and a 90-second four-client load (57 complete generations, 14,592 tokens). The **156.51 tokens/s** clipped client-window model and **156.1388 tokens/s** full-drain rate use different denominators. These load numbers are retained observations, not a new serving-capacity qualification.

Needle checks establish presence/parity, with continued role/thinking markers. Client closure and exact follow-up do not prove backend cancellation retirement. Eager mode uses configuration plus pinned source semantics, not a kernel trace. Full fixture request bodies/seed are unavailable. Switched/single-rail hardware, GPU mixed-profile rejection, allocated relay, TP1/2/3, Flash Next, production peer loss, full-window quality, vision, long-duration reliability, reboot, tailnet inference and the private-assistant role remain unqualified. Unpublished upstream fixtures remain unavailable; no absolute-speed reproduction or Python-versus-Zig claim.

The former 471-file directory (README plus 470 evidence files) is preserved at immutable [archive commit 921ce7e](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/921ce7e7ba8ce7ad6767fb0d8a99ef5c05a88cf1/docs/tp4-rdma-focused), also tagged `evidence/tp4-rdma-qualified-20261022`. It contains the original build receipts, fixture JSONL, synthetic requests/responses, probes, logs, manifest and standalone verifier. Each compact JSON identifies exact archived paths and original-byte SHA-256 values. Summary schemas are labeled as extractions; original schema/version fields are retained. [Source provenance](../../zig/TRANSPORT_SOURCE.json) remains in this change. No production code or tested binary is changed by compacting the evidence.

For later offline receipt reproduction, use a checkout of the archive commit above, then run the archived command (not run during this cleanup):

```sh
cd docs/tp4-rdma-focused/gpu
python3 verify.py --receipts . --out /tmp/glm-tp4-analysis.json
```

That verifier reproduces the archived short-workload comparison and checks retained receipt hashes/parity; it does not reproduce the sustained-load analyzer, whose source is hash-pinned but absent from the archive. Native source reconstruction instructions and exact source hashes remain in `zig/TRANSPORT_SOURCE.json`; hardware qualification requires its own authorized gates.

Credit: **BertholomusAI (Albert Lee, @bertholomus)** for the registered-memory/DeepSeek RDMA work and benchmark client; existing CREDITS/NOTICE remain authoritative.
