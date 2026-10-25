# Registered TP4 ring: qualification and performance, 2026-10-25

The same `a6645bea` engine passed the four-rank NCCL and registered-RDMA probes and GLM correctness checks, then the matched timing matrix below. Exact source, binary, image and model pins are in [qualification.json](qualification.json); measured waves, counts and source hashes are in [performance.json](performance.json). Full logs remain in the private evidence archives rather than this commit.

The native CUDA profile uses GLM checkpoint `078455ff` and DFlash2 `bf582e4e`, `--context 131072 --parallel 8 --max-tokens 32768 --prompt-cache-gib 3 --thinking`, L2 prefetch and shared-expert side stream enabled. The physical rank order is spark → spark-peer → b9dc → c21f, GID3, with a 512 KiB registered-relay cutoff and the pinned patched NCCL 2.30.7 for fallback. The image fingerprint and engine bytes match across all four hosts. Requests go directly to the head's administrative loopback `127.0.0.1:40053`; client gateways stay in maintenance.

The qualification includes 64 changing-payload mixed-graph replays per rank and mode, byte-exact checks across the cutoff, 31 settled correctness requests per mode, 12 matching cross-mode token comparisons, both-key tool checks and an eight-client gateway smoke. Canonical, qualification-build and final engine SHA-256 values are equal. These are bounded correctness checks, without full-window quality or deployment acceptance.

Timing ran sequentially NCCL then registered RDMA. Each profile uses two fresh model lifetimes, with one warmup and three measured waves for each prompt/concurrency cell. A separate NCCL preparation lifetime preceded measurement. All **312 requests** were valid: **117 measured + 39 warmup per mode**, with exactly 400 output tokens each. All 156 paired output-token hashes match, including 117 measured requests; this is an observed parity count. Each measured lifetime reached HTTP200 within 79 s and all 16 compared rank containers exited 0 without OOM.

| Prompt target (actual) | C | NCCL tok/s | RDMA tok/s | RDMA change | NCCL TTFT ms | RDMA TTFT ms | Comparison |
|---|---:|---:|---:|---:|---:|---:|---|
| 256 (256–256) | 1 | 58.60 | 68.54 | 16.95% | 349.05 | 328.75 | complete |
| 256 (256–256) | 4 | 110.97 | 129.29 | 16.51% | 863.48 | 841.87 | complete |
| 256 (256–256) | 8 | 145.90 | 161.16 | 10.46% | 1552.70 | 1481.82 | complete |
| 4096 (4096–4096) | 1 | 44.22 | 49.62 | 12.23% | 2554.78 | 2511.03 | complete |
| 4096 (4096–4096) | 4 | 69.30 | 75.82 | 9.40% | 6364.03 | 6267.40 | complete |
| 4096 (4096–4096) | 8 | 81.39 | 86.08 | 5.76% | 11475.36 | 11321.43 | complete |
| 32768 (32768–32768) | 1 | 15.67 | 16.43 | 4.85% | 19311.04 | 19100.24 | complete |
| 32768 (32768–32768) | 4 | 17.75 | 18.32 | 3.20% | 48308.35 | 47750.43 | complete |
| 32768 (32768–32768) | 8 | 18.53 | 18.89 | 1.96% | 86782.38 | 85881.96 | complete |

Prompt lengths are exact tokenizer counts; C is client concurrency. Each throughput value is the median of three aggregate wave rates, `400 × C / (last completion − first dispatch)`, including TTFT and returned-token-ID overhead. TTFT is the median time to first nonempty content event. Request sampling uses temperature 0, top-p 1, request thinking disabled and draft enabled. All nine cells are complete.

**RDMA throughput was 1.96%–16.95% higher in these samples. NCCL precedes RDMA, so order is confounded with time.** Cache usage is observed rather than reset. Three waves per cell support a descriptive comparison, without a statistical significance, p95, quality, simultaneous-backend-decoding, full-context, sustained-capacity, reboot or reliability claim.

The comparison uses the four completed lifetimes listed in the receipt. Timestamped cleanup checks cover the entire operation and preserve all prior containers plus 24 operation containers. At 2026-10-25T18:31:35.153526+00:00, all GPUs were idle, model listeners absent, all eight fleet locks released and all three gateways in maintenance (both keys 503; unauthenticated 401). No prior model was restored. Later live state requires a fresh audit.

Private raw benchmark archive: `/Users/swb/Documents/ChatGPT/spark-evidence/glm-pr204-bench-20261025`; earlier qualification logs remain at local recipe commit `70407067440f6ec1c0384577dbc1bf1bfd96662d`, path `docs/tp4-registered-relay-20261025`, and Spark commit `c83957e34618ea2c45399dc4c57de87350c155bb`. The small receipts retain immutable raw-file hashes. The earlier qualification's caffeinate label tracked a launcher parent; actual inhibitor-PID monitoring was corrected and power assertions observed for the timing run. Historical receipts remain intact.
