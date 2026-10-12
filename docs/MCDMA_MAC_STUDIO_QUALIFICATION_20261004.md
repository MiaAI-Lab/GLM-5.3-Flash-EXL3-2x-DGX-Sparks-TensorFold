# Independent Mac Studio / MCDMA qualification

Date: 2026-10-04

This note records an independent experiment built from Mia's AI Lab's two-Spark
GLM-5.3-Flash recipe. It is not a supported Mia recipe mode and does not change
this repository's defaults, patches, images, or published measurements.

## Question

Can two DGX Sparks prefill a request, transfer the resulting state over MCDMA,
and let an M3 Ultra Mac Studio decode quickly enough to beat the published
two-Spark one-request result?

The independent implementation and retained evidence are in
[`spenchey/GLM-5.3-Flash-MCDMA-2x-DGX-Sparks-Mac-Studio`](https://github.com/spenchey/GLM-5.3-Flash-MCDMA-2x-DGX-Sparks-Mac-Studio).
That repository credits this recipe, TensorFold, MCDMA, the model authors, and
the other upstream work it uses.

## Frozen comparison

The candidate used Mia's C1 prose prompt and request shape: one request,
thinking off, greedy generation, a 32-token warmup, and 400 measured output
tokens. Five cold runs were taken after a warm canary.

| Measure | Published two-Spark target | Three-machine candidate |
| --- | ---: | ---: |
| Decode | 60.4 tok/s | 61.270 tok/s |
| TTFT | 0.170 s | 0.397 s |
| Implied / measured complete time | 6.776 s | 6.907 s |

The three-machine decoder was about 1.4% faster, but its additional prefill and
handoff time made the complete request about 1.9% slower. It did not meet the
experiment's stricter 3% end-to-end win gate (6.573 s or less).

The Spark side used the v1.5 image and patch label recorded in the independent
receipt. The Mac used TensorFold 0.6.5 plus the experiment's reviewed GLM MLX
stage patch. The candidate and Mia's published service used different
checkpoints, so this is a systems-performance comparison against the published
target, not a model-quality or byte-identical-output comparison between them.

## Transport result

- The prompt-cache state crossed the physical MCDMA path; counters advanced on
  both ends with zero transfer failures.
- Both Spark containers stayed out of OOM state.
- The candidate's own Spark-prefill and Mac-prefill paths agreed through the
  retained warm canary, but the full candidate still has an unresolved token
  divergence at output token 104 against its local control.
- MCDMA push versus pull and the current Mia image changed the median by only a
  few milliseconds; neither closed the end-to-end gap.
- All owned Spark, Studio, and MCDMA processes were stopped after the campaign.

## Reproduce and inspect

Start with the independent repository's README and pinned configuration. The
most relevant evidence is:

- `results/receipts/2026-10-04-current-mia-image-candidate.md`
- `results/receipts/2026-10-04-protocol4-push-pull.md`
- `docs/GOAL-SINGLE-STREAM-MCDMA.md`

Raw logs remain excluded because they can contain private host details and
memory-region credentials. The public receipts retain their hashes and the
exact software identities needed to audit the result.

## Conclusion

The MCDMA path is functional and the Mac decoder can slightly exceed the
published two-Spark decode rate, but the complete request is still slower. The
honest next target is reducing TTFT and handoff overhead while resolving the
candidate's output divergence; the current result should not replace this
recipe's two-Spark baseline.
