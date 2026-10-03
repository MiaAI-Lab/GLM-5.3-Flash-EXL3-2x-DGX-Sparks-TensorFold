# Hermes Agent and the optional abliterated checkpoint

The default checkpoint remains Mia's standard quantization. An independently derived abliterated checkpoint can also serve on the v1.5 TensorFold v0.6.0 image. Do not identify its weights as Mia's standard checkpoint.

## Explicit checkpoint identity

After configuring the pair as described in the main README, set these overrides in your own shell or `scripts/local.sh`:

```bash
export MODEL_ID=bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit
export MODEL_REVISION=14858211ed81d7fa773f8a0db02f38f36d230252
export SERVED_NAME=GLM-5.3-Flash-EXL3-Abliterated
export HOST=127.0.0.1
export CONTEXT=1048576
export PARALLEL=4
export MEMORY_RESERVE_GIB=18.5
./scripts/prepare.sh
./start.sh
```

`MODEL_REVISION` is explicit because a pin for the standard repository must never be silently applied to another repository. `SERVED_NAME` makes API responses and monitoring distinguish the variant. Keep both ranks on the same immutable runtime, target revision and drafter revision. This does not train, edit or redistribute weights.

For standard GLM, use `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` at `6c5b28260ab9e80c6608de8b419e624cbe71b7cf`, with `SERVED_NAME=GLM-5.3-Flash-EXL3`. Relative to the prior pin, only LICENSE, LICENSE-GLM-5.3-Flash and README.md metadata changed; weight/config/tokenizer blob identities are unchanged. Refresh snapshot verification for the new revision rather than renaming an old receipt.

The measured drafter is `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`. Check its separate license and the target's license for your intended use. These measurements were private, personal/noncommercial tests, not approval for a commercial deployment.

## Hermes custom endpoint

Use `hermes model`, choose Custom endpoint, and point it at the loopback TensorFold API. Equivalent model settings in the selected profile's `config.yaml` are:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8888/v1
  default: GLM-5.3-Flash-EXL3-Abliterated
  context_length: 1048576
```

Use the standard served ID instead when running standard weights. Verify `/v1/models` and the model field of actual replies rather than trusting a requested alias. Keep the endpoint loopback/private; use an authenticated private transport for another machine. An unauthenticated OpenAI-compatible API must not be exposed publicly.

The context value is configured capacity, not a full-window stress qualification. Native tool calling, incremental SSE, reasoning and a synthetic image were checked through an OpenAI-compatible client. This is not a claim that every Hermes toolset or auxiliary-vision routing choice was independently tested. Current configuration guidance: https://hermes-agent.nousresearch.com/docs/integrations/providers/.

## Reproduce the synthetic timings

Use an idle endpoint and arrange a bounded exclusive test window outside the harness. It never stops/restarts engines or changes services. It rejects an unexpected served model; `--exclusive` also rejects telemetry showing overlapping inference.

```bash
python3 tools/hermes_benchmark.py --model GLM-5.3-Flash-EXL3-Abliterated --exclusive > abliterated.jsonl
python3 -m unittest discover -s tests -p 'test_hermes*.py'
```

Run the same prompts, reply budgets and runtime settings against standard weights for comparison. Preserve cold/warm prefix-cache distinctions. The output includes synthetic generated code for local quality testing and full health snapshots; inspect/redact it before sharing. Do not substitute chat histories or customer prompts into a public report.

See [measured results](hermes-benchmark-results.md) for the workload, definitions, limits and exact pins.
