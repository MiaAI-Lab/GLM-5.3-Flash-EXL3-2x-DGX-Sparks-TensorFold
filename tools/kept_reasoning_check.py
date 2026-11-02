"""CPU check of patch 0121: KeptReasoning.restore in one pass over the conversation, the same restores as before.

  python3 -B tools/kept_reasoning_check.py --source-root /path/to/patched/src
Use --expect-stock on the unpatched source to reproduce the cost: restore() re-encodes and hashes the whole
conversation before every assistant tool step that came back without reasoning, so one request costs the square of
its length (seconds at ~2,000 messages, with the GIL held).
Loads tensorfold/families/glm5_next/cuda/kept_reasoning.py by its path (no torch, GPU or model) and compares it with
the stock implementation, embedded below as the oracle, on random agent conversations: the same restored messages,
the same copy-or-not, LRU eviction included.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ the stock implementation (patch 0036), the oracle
_OUR_ID = re.compile(r"^call_[0-9a-f]{24}$")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value) if value.strip() else {}
        except ValueError:
            return value
    return {} if value is None else value


def _calls(calls: Any) -> list[list[Any]]:
    out = []
    for call in calls if isinstance(calls, list) else []:
        fn = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
        out.append([str(fn.get("name") or ""), _arguments(fn.get("arguments"))])
    return out


def _signature(history: list[Any], calls: Any) -> str:
    shape = []
    for m in history:
        if not isinstance(m, dict):
            continue
        entry = [str(m.get("role") or ""), _text(m.get("content")).strip()]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            entry.append(_calls(m.get("tool_calls")))
        shape.append(entry)
    text = json.dumps([shape, _calls(calls)], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "sig:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _keys(history: list[Any], calls: Any) -> list[str]:
    ids = [c.get("id") for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
    return [i for i in ids if isinstance(i, str) and _OUR_ID.match(i)] + [_signature(history, calls)]


class StockKeptReasoning:
    def __init__(self, entries: int = 1024, chars: int = 32 << 20) -> None:
        self.entries, self.chars = int(entries), int(chars)
        self.items: OrderedDict[int, tuple[str, tuple[str, ...]]] = OrderedDict()
        self.by_key: dict[str, int] = {}
        self.size = self.next = 0
        self.lock = threading.Lock()

    def remember(self, messages: Any, calls: Any, reasoning: Any) -> None:
        if not calls or not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning) > self.chars:
            return
        keys = tuple(_keys(messages if isinstance(messages, list) else [], calls))
        with self.lock:
            n, self.next = self.next, self.next + 1
            self.items[n] = (reasoning, keys)
            self.size += len(reasoning)
            for key in keys:
                self.by_key[key] = n
            while len(self.items) > self.entries or self.size > self.chars:
                old, (text, old_keys) = self.items.popitem(last=False)
                self.size -= len(text)
                for key in old_keys:
                    if self.by_key.get(key) == old:
                        del self.by_key[key]

    def _find(self, keys: list[str]) -> str | None:
        with self.lock:
            for key in keys:
                n = self.by_key.get(key)
                if n is not None:
                    self.items.move_to_end(n)
                    return self.items[n][0]
        return None

    def restore(self, messages: Any) -> Any:
        if not isinstance(messages, list) or not self.items:
            return messages
        out = None
        for i, m in enumerate(messages):
            if not (isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
                    and not _text(m.get("reasoning_content")).strip() and not _text(m.get("reasoning")).strip()
                    and "</think>" not in _text(m.get("content"))):
                continue
            kept = self._find(_keys(messages[:i], m["tool_calls"]))
            if kept is not None:
                out = out if out is not None else list(messages)
                out[i] = {**m, "reasoning_content": kept}
        return messages if out is None else out


# ------------------------------------------------------------------ the check
def load(root: str):
    path = Path(root) / "tensorfold/families/glm5_next/cuda/kept_reasoning.py"
    spec = importlib.util.spec_from_file_location("kept_reasoning_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conversation(r: random.Random, steps: int):
    """An agent conversation as a client sends it back, and the replies the server saw (messages before, calls, reasoning)."""

    msgs: list[Any] = [{"role": "system", "content": "system " + str(r.random())}]
    replies = []
    for s in range(steps):
        if r.random() < 0.08:
            msgs.append({"role": "user", "content": f"user {s} {r.random()}"})
        if r.random() < 0.02:
            msgs.append("not a message")                 # skipped by the shape
        cid = f"call_{r.getrandbits(96):024x}" if r.random() < 0.5 else f"toolu_{s}"
        calls = [{"id": cid, "type": "function",
                  "function": {"name": r.choice(["read", "bash"]), "arguments": json.dumps({"p": s, "q": r.random()})}}]
        reasoning = r.choice(["", " ", f"thinking {s} {r.random()}"])     # empty think blocks are common in tool loops
        replies.append((copy.deepcopy(msgs), calls, reasoning))
        sent = {"role": "assistant", "content": r.choice(["", "ok", None]), "tool_calls": copy.deepcopy(calls)}
        mode = r.random()
        if mode < 0.15 and reasoning.strip():
            sent["reasoning_content"] = reasoning         # a client that sends it back
        elif mode < 0.25:
            sent["tool_calls"][0]["id"] = f"renumbered_{s}"   # a client that renumbers ids: found by the signature
        elif mode < 0.30:                                 # re-serialised arguments: the same signature
            args = json.loads(calls[0]["function"]["arguments"])
            sent["tool_calls"][0]["function"]["arguments"] = json.dumps(args, indent=1)
        msgs.append(sent)
        msgs.append({"role": "tool", "tool_call_id": sent["tool_calls"][0]["id"], "content": f"result {s} " * 50})
    return msgs, replies


def drive(cls, msgs, replies, entries: int):
    kept = cls(entries, 32 << 20)
    for before, calls, reasoning in replies:
        kept.remember(before, calls, reasoning)
    return kept.restore(msgs)


def timed_restore(cls, msgs, replies) -> float:
    kept = cls()
    kept.remember(replies[0][0], replies[0][1], "seed")  # a non-empty store, as on a live server
    bare = [({k: v for k, v in m.items() if k != "reasoning_content"} if isinstance(m, dict) else m) for m in msgs]
    t0 = time.perf_counter()
    kept.restore(bare)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    mod = load(args.source_root)

    r = random.Random(7)
    restored = 0
    for trial in range(300):
        msgs, replies = conversation(r, r.randint(1, 60))
        entries = r.choice([1024, 8, 3])                 # small LRUs exercise eviction
        want = drive(StockKeptReasoning, msgs, replies, entries)
        got = drive(mod.KeptReasoning, msgs, replies, entries)
        assert got == want, f"trial {trial}: restore differs from stock"
        assert (got is msgs) == (want is msgs), f"trial {trial}: copy semantics differ from stock"
        restored += sum(1 for a, b in zip(msgs, got) if a is not b)
    print(f"300 conversations, {restored} steps restored: the same restores as stock")

    msgs, replies = conversation(random.Random(1), 1000)   # ~2,100 messages: a long agent session
    stock = timed_restore(StockKeptReasoning, msgs, replies)
    here = timed_restore(mod.KeptReasoning, msgs, replies)
    print(f"restore over {len(msgs)} messages: stock {stock:.3f} s, this source {here:.3f} s")
    if args.expect_stock:
        assert here > stock / 4, "this source restores in one pass: is it patched?"
        print("stock: restore re-encodes the conversation before every step (the square of its length)")
        return 0
    assert here * 20 < stock, "restore is not one pass over the conversation"
    print("kept reasoning: one pass a request, the same restores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
