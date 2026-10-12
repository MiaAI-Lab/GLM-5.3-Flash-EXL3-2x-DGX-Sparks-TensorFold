# Literal tool closers in GLM arguments

Patch 0110 shares argument-aware envelope boundaries between the CUDA server's
content filter, final tool parser, and thinking/tool split. A literal
`</tool_call>` inside a GLM `<arg_value>` is argument data until its
`</arg_value>` delimiter, not the end of the enclosing call.

Previously, a streaming write could emit both a correct structured call and
its argument tail as ordinary assistant content. Non-streaming final parsing
could also truncate the arguments. Partial values now stay hidden, and the
single-call response policy uses the same CUDA boundary filter first.

This does not change model tokens, sampling, scheduling, caches, or tool
execution. It does not make literal `</arg_value>` inside an unescaped GLM
value unambiguous, or remove intentional examples from reasoning text.

Run the CPU regression against the patched image:

```sh
python tests/tool_envelope_contract.py
```

It exercises the actual `App.run` response path using a fake character-token
engine: fragmented boundaries, thinking on/off, streaming/non-streaming,
single/multiple calls, truncation, nested literal tool markup, and JSON/Qwen
final-parser fallback. No GPU, weights, network, or executed tools are needed.
