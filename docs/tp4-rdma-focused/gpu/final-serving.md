## Final serving receipt audit

- Sustained client window:90seconds, four clients,57 completed requests,14592 completion tokens, zero worker/metadata errors. Observed peak client in-flight requests:4.
- Pinned summary aggregate156.51tok/s models token contributions over the clipped90-second decode window. Raw full-drain span 93.455328s yields 156.138770tok/s and 0.609917requests/s; these use a different denominator.
- Summary per-request decode rates p05/p50/p95:26.804/40.292/76.019tok/s; TTFT p50/p95:0.182/0.302s.
- Supplemental acceptance:31/31 settled cases,18 completed generations,12 tokenizer sizing calls, one intentional client cancellation;12/12 exact token comparisons verify from raw SSE.
- Needle:32746raw tokens and32752chat-template prompt tokens; required value recalled in all four outputs. Both cache-reuse calls report32704cached tokens. Continued output includes visible role/thinking markers; no clean exact-answer quality claim.
- Cancellation:connection closes after first content; following short serial/draft pair is exact. Backend retirement timing remains unverified.
- Eager/graph:all4 immutable container configs show TF_GLM_EAGER=1 versus absent, with matching per-host image/command hashes. Presence-based pinned source semantics, not a kernel execution trace.
- Overall offline serving receipt validation:True. Errors:load=[]; acceptance=[].
