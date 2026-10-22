# Focused TP4 NCCL / registered RDMA receipts

Offline verification passed using the standalone [verify.py](verify.py). The machine-readable result is [analysis.json](analysis.json).

The unchanged Grill comparison records **+16.9938889634%** observed throughput change, with nominal 95% model-based interval **[+15.7928411428%, +18.2073945133%]**. The validator independently reproduces the effect and interval from eight acquisition medians per transport.

| Receipt | NCCL | Registered RDMA |
| --- | ---: | ---: |
| Complete acquisitions | 8 | 8 |
| Requests, including warmup | 32 | 32 |
| Measured requests | 24 | 24 |
| Reported completion tokens | 12,800 | 12,800 |
| Median of acquisition medians, tokens/s | 157.0866956153 | 184.4140233466 |

Each acquisition contains one warmup and three measured waves, with one concurrent request and exactly 400 completion tokens per request. The request bytes match across both captures. The effect is the exponential of the difference of mean log acquisition medians; it is not the percentage ratio of the two table medians. The interval uses the log-scale heteroscedastic standard error, critical value 2.364624251 and conservative seven degrees of freedom.

The validator checks the full retained capture trees, workload and deployment hashes, plan/session/reservation links, each request hash, each raw SSE response hash and byte count, terminal and usage records, wave throughput, acquisition medians, and the official report's hashes, accounting and statistics. Capture-tree identities use SHA256 over sorted relative paths, a NUL, each file's hexadecimal SHA256 and a newline; all 282 files are covered.

All three qualification profiles pass with 89 rows and exactly 40 completions each. NCCL, registered RDMA and eager completions match in every field except client timing fields. This includes output text, returned token IDs, hashes, case/key identities, prompt hash, requested output dimensions, draft mode, finish reason, usage and cache dimensions. Serial/draft parity covers rendered prompt lengths 1023/1024/1025, 1535/1536/1537 and 2050/2051/2052; eight solo/overlap pairs also match.

The three [projected qualification receipts](captures/) preserve row order and every field value except the removed top-level `backend` field. [derivation.json](derivation.json) records each original byte SHA256, output byte SHA256, row count and field-exact projection check. Original identities are recorded by digest without requiring a separate presentation label in the receipt API.

Current final acceptance also passes: the candidate ID and eager/graph receipts match their source hashes; 98 referenced current receipt files match their SHA256 and byte counts; all 31 settled requests have independently checked request and response hashes; and all 12 declared pairs have matching request dimensions and exact output tokens/text. The cancellation follow-up ordering and needle-value presence also pass. The retained [final-serving.json](final-serving.json) remains the source for serving-load figures; this validator does not recompute that load analysis or execute its collector/analyzer.

From the `gpu` receipt directory:

```sh
python3 verify.py --receipts . --out /tmp/glm-tp4-analysis.json
```

The receipts describe sequential capture periods. Drift, carryover and correlation can make the nominal interval too narrow. They establish no causal attribution, equivalence, noninferiority, guaranteed precision, general concurrency result, full-context quality or long-duration reliability. Serving identities and settings in performance declarations are not runtime attestation.

Qualification JSONL receipts omit request bodies and seed, so their exact replay cannot be reconstructed from prompt hashes alone. Rendered prompt lengths do not independently attest internal prefill widths, and overlapping client intervals do not prove simultaneous backend execution. Needle presence/parity does not establish clean answer quality. Closing a client SSE connection and using the service afterward does not establish backend cancellation retirement. No live, network, GPU, build or reboot check was run for this verification.
