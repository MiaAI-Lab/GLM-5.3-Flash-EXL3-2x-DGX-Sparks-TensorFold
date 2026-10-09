# Hermes Agent

[Hermes Agent](https://hermes-agent.nousresearch.com/) (Nous Research) can use this server as a custom endpoint. The
server is OpenAI-compatible, so no recipe change is needed. Hermes setup and the synthetic timing harness here were
written by [Steve Darlow (@kerpopule)](https://github.com/kerpopule), who authored the work in PR #52.

## Hermes custom endpoint

Use `hermes model`, choose Custom endpoint, and point it at the TensorFold API. Equivalent model settings in the
selected profile's `config.yaml`:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8888/v1
  default: GLM-5.3-Flash-EXL3
  context_length: 1048576
```

`default` is `SERVED_NAME` (`GLM-5.3-Flash-EXL3` unless changed). Verify `/v1/models` and the model field of actual
replies rather than trusting a requested alias. Keep the endpoint loopback or private (`HOST=127.0.0.1`); an
unauthenticated OpenAI-compatible API must not be exposed publicly. The context value is configured capacity, not a
full-window stress qualification. Current Hermes guidance:
https://hermes-agent.nousresearch.com/docs/integrations/providers/.

## Reproduce the synthetic timings

Use an idle endpoint and arrange a bounded exclusive test window outside the harness. It never stops or restarts
engines. It rejects an unexpected served model; `--exclusive` also rejects telemetry showing overlapping inference.

```bash
python3 tools/hermes_benchmark.py --model GLM-5.3-Flash-EXL3 --exclusive > run.jsonl
python3 -m unittest tests/test_hermes_benchmark.py
```

The output includes synthetic generated code and full health snapshots; inspect it before sharing.
