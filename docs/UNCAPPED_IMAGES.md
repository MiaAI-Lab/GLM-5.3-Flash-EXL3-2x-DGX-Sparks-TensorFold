# Optional image-count and shared-detail limits

Patch 0109 allows `TENSORFOLD_GLM_MAX_IMAGES=0` and
`TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS=0`, independently. Existing nonzero defaults
are unchanged. Zero count permits image-bearing histories to grow until another
resource or context limit applies. Zero shared token budget retains each image's
configured allowance instead of dividing 16,384 tokens between all images.

The count is a serving policy, not an architectural 50-image restriction: the
live checks below demonstrate successful requests with 51 images. They do not
establish that arbitrary image counts or sizes fit in memory or context.

## Live evidence (2026-10-13)

Two independent three-node DGX Spark deployments used the recipe at
`3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab` plus this patch, TensorFold 0.6.0,
GLM-5.3-Flash EXL3 4bpw, DFlash2, BF16 dense and KV, 262,144 total context,
four request slots, Max thinking, temperature 1, top-p 0.95 and a 32,768-token
output allowance. Model revision:
`078455ffe6472f9a52fbc1139f58b9db2881b25c`.

Each request contained 51 identical 64x64 red PNG data URLs and asked for their
common dominant color. Each was repeated once. All four responses said `Red`.

| Deployment | First request | Repeated request | Cached tokens on repeat |
| --- | ---: | ---: | ---: |
| A | 2.552 s | 1.395 s | 896 |
| B | 2.551 s | 1.422 s | 896 |

Each request used 945 prompt tokens and 49 completion tokens (47 reasoning).
First requests reported zero cached tokens. These are end-to-end timings for
this small fixture, not a general speed benchmark. Identical small pictures
exercise count admission and cache reuse, not distinct-image understanding or
full-resolution capacity. No quality advantage for large image collections is
claimed from this fixture.

The unchanged per-image token cap and opt-out of count-based downscaling are
also checked directly by CPU tests. No GPU test established that 51 images at
their maximum token allowance are safe at every concurrency/context setting.

## Retained limits

- Per-image encoded bytes, dimensions and pixels remain bounded.
- Uncapped GLM count retains the default 50-image aggregate budgets: 64 MiB
  encoded bytes, 400 MiPixels and 100 seconds for image loading.
- Cached pictures count toward aggregate pixels as well as bytes/time.
- Per-image token allowance and the server's total-context checks remain.
- Video limits and all nonzero image-count behavior remain, except that cached
  pictures can no longer bypass the aggregate pixel check.

Removing count-based downscaling can increase image tokens, prefill work and
memory. Deployments must size their context and concurrency accordingly; a
request can still be rejected by other limits.

## CPU regression checks

Apply the recipe patches in order to TensorFold 0.6.0, then run:

```sh
python3 tools/test_uncapped_images.py --source-root /path/to/patched/source
```

Six checks passed: unchanged defaults and opt-in token allowances; image-count
admission and loading; zero rejected for resource budgets; equality of retained
aggregate budgets; encoded-byte enforcement; cached-image pixel enforcement.
Requires numpy and Pillow, with no GPU, model download or server.
