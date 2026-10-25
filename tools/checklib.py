"""What the port's end-to-end checks share (spill_check.py, resume_bench.py, prefix_fanout.py): a server client that
returns strong token hashes, streaming with cancellation and inter-token gaps, the engine's counters from /health and
/metrics as one flat dict, deterministic prose sized to the window, file operations on a rank's spill directory (here,
over a shell prefix such as `ssh <worker>`, optionally under sudo), and a receipt of per-check PASS / FAIL / SKIP
results. From the DeepSeek-V4.1-Flash recipe's port (wojo's fork, tools/checklib.py), for GLM-5.3-Flash on the Zig
engine (ENGINE=zig): its served name, its window, its API key variable and none of DeepSeek's KV-shard field lists.

Stdlib only. API key: the API_KEY environment variable (its 0/1 switch values are not keys), else TENSORFOLD_API_KEY
(the variable this recipe passes to the server; the first of its comma-separated keys), else the file named by a tool's
--api-key-file flag (add_api_key_arg / use_api_key_args). Nothing else is read: no default path, no API_KEY_FILE
variable, no home folder (this recipe writes no key file). The key goes into the Authorization header only; receipts
and printed lines are scrubbed of it (redact). Proxies are never used.
Sizes: the tools' long prompts are capped to the window (CONTEXT, 131,072 tokens by default for ENGINE=zig; fit()).
Counter names are the DeepSeek port's cache-tier names (its core/prompt_cache.zig TierCounts), which the GLM port's
cache core re-implements on TensorFold v1.0.7 (check them against the engine once it is in zig/patches): /health
`prompt_cache{...}` with `spill` an ARRAY of per-rank objects `{rank, <TierCounts>}`, /metrics
`tensorfold:prompt_cache_*` and `tensorfold:spill_<name>{rank="N"}`. Flat keys are `prompt_cache.<name>`; for the tier,
`spill.<name>.rank<N>` per rank and `spill.<name>` over the ranks (the sum; the minimum for disk_free_bytes and
cap_bytes, where the tightest rank binds); plus the top-level `chats_continued`, `chats_continued_prefill_tokens`. KV
sharding's counters are read as they come: /health `kv_shard{..., per_part: [{part, ...}]}` and /metrics
`tensorfold:kv_shard_<name>`, `tensorfold:kv_shard_part_<row>{part="N"}` as `kv_shard.<name>`, `kv_shard.part<N>.<row>`.
"""
import hashlib
import http.client
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_MODEL = "GLM-5.3-Flash-EXL3"         # SERVED_NAME (scripts/config.sh)
DEFAULT_CONTEXT = 131072                     # CONTEXT for ENGINE=zig (scripts/config.sh): the captured kernels' arena
FIT_SHARE = 0.9                              # fit(): the share of the window a prompt may take (TOKENS_PER_WORD's error)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
TOKENS_PER_WORD = 1.15      # the DeepSeek tokenizer's on WORDS prose, not measured on GLM's: exact.py calibrates
                            # against usage.prompt_tokens, and fit() leaves the rest of the window for the difference


def log(*m):
    print(*m, file=sys.stderr, flush=True)


def sha(s) -> str:
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


def default_url() -> str:
    return os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888"))


def default_context() -> int:
    """The window the tools size prompts for: CONTEXT (the recipe's setting), else 131,072 (ENGINE=zig's default)."""
    try:
        return int(os.environ.get("CONTEXT") or DEFAULT_CONTEXT)
    except ValueError:
        raise SystemExit(f"CONTEXT is a number of tokens, not {os.environ.get('CONTEXT')!r}")


def fit(context: int, reply: int = 0) -> int:
    """The largest prompt (tokens) a tool asks for in a window of `context`: FIT_SHARE of it, less the reply."""
    return max(1, int(context * FIT_SHARE) - reply)


def capped(tokens: int, context: int, reply: int = 0, what: str = "a prompt") -> int:
    """tokens, at most fit(context, reply); says so when it cuts."""
    top = fit(context, reply)
    if tokens > top:
        log(f"{what}: {tokens} tokens capped to {top} (the window is {context}: CONTEXT or --context)")
        return top
    return tokens


def over_window(sizes, context: int, reply: int = 0) -> list:
    """The sizes that do not fit the window (fit(context, reply))."""
    return [s for s in sizes if s > fit(context, reply)]


_api_key_file = None    # set only by use_api_key_args (a tool's --api-key-file): never a default path


def add_api_key_arg(ap):
    """The one way a tool names a key file: --api-key-file PATH (the server's API_KEY_FILE, e.g. $STATE_DIR/api-key)."""
    ap.add_argument("--api-key-file", metavar="PATH", default=None,
                    help="read the server's API key from this file (first non-comment line); default: the API_KEY "
                         "environment variable, else no key")
    return ap


def use_api_key_args(a):
    """Take a parsed --api-key-file (None leaves only API_KEY)."""
    global _api_key_file
    _api_key_file = getattr(a, "api_key_file", None)
    return a


def api_key() -> str:
    """API_KEY from the environment (its 0/1 switch values are not keys), else the first of TENSORFOLD_API_KEY's
    comma-separated keys (as the server reads them), else the --api-key-file file, else none."""
    key = os.environ.get("API_KEY", "").strip()
    if key and key not in ("0", "1"):
        return key
    key = next((k.strip() for k in os.environ.get("TENSORFOLD_API_KEY", "").split(",") if k.strip()), "")
    if key:
        return key
    if not _api_key_file:
        return ""
    try:
        with open(_api_key_file) as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError as e:
        raise SystemExit(f"--api-key-file {_api_key_file}: cannot read it ({e.strerror})")
    return line.split(":", 1)[1].strip() if ":" in line else line


def redact(v):
    """v (a string, or lists / dicts of them) with the API key, when there is one, replaced by <api-key>."""
    k = api_key()
    if not k:
        return v
    if isinstance(v, str):
        return v.replace(k, "<api-key>")
    if isinstance(v, list):
        return [redact(x) for x in v]
    if isinstance(v, dict):
        return {redact(x): redact(y) for x, y in v.items()}
    return v


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    k = api_key()
    if k:
        h["Authorization"] = "Bearer " + k
    return h


_units = {}
_units_lock = threading.Lock()


def words(n: int, seed: int) -> str:
    """The first n words of a deterministic prose stream (prefix-stable: words(n) is a prefix of words(n+1))."""
    with _units_lock:
        if seed not in _units:
            _units[seed] = (random.Random(seed), [])
        rng, units = _units[seed]
        while len(units) < n:
            k = rng.randint(6, 16)
            s = [rng.choice(WORDS) for _ in range(k)]
            s[-1] += "."
            units.extend(s)
        return " ".join(units[:n])


def words_for(tokens: int) -> int:
    return max(1, int(tokens / TOKENS_PER_WORD))


# The DeepSeek port's core/prompt_cache.zig TierCounts, in its order (the per-rank disk-tier counters), without its
# picture points' two (GLM's lane host keeps no picture prompts): the names the GLM cache core is expected to report.
SPILL_FIELDS = ("points blocks stored_bytes cap_bytes disk_free_bytes staging_bytes staging_peak writes_pending "
                "writes_queued writes_committed writes_cancelled writes_cancelled_guard writes_cancelled_staging "
                "writes_cancelled_shutdown writes_failed forced_staging no_space dedup_blocks dedup_bytes written_bytes "
                "presence_refused compatible_misses crc_errors size_errors reads read_bytes read_failed evicted_turns "
                "evicted_forks evicted_conversations evicted_shared quota_trims recovered_points "
                "dropped_temporaries").split()
SPILL_GAUGES = ("points", "blocks", "stored_bytes", "cap_bytes", "disk_free_bytes", "staging_bytes", "staging_peak",
                "writes_pending")
SPILL_MIN = ("disk_free_bytes", "cap_bytes")       # aggregated as the minimum over the ranks, not the sum
SPILL_MAX = ("staging_peak",)                      # each rank's own peak against its own cap: the maximum, not the sum



def put_spill_rank(out: dict, rank: int, name: str, value):
    out[f"spill.{name}.rank{rank}"] = value


def aggregate_spill(out: dict):
    """spill.<name> from every spill.<name>.rank<N> (sum, min for SPILL_MIN, max for SPILL_MAX); leaves existing
    aggregates alone."""
    per = {}
    for k, v in out.items():
        if k.startswith("spill.") and ".rank" in k:
            name, _, r = k[len("spill."):].rpartition(".rank")
            if r.isdigit():
                per.setdefault(name, []).append(v)
    for name, vs in per.items():
        out.setdefault("spill." + name, min(vs) if name in SPILL_MIN else max(vs) if name in SPILL_MAX else sum(vs))


def rank_values(c: dict, name: str) -> dict:
    """{rank: value} of a tier counter (empty when the server reports none per rank)."""
    out = {}
    pre = f"spill.{name}.rank"
    for k, v in c.items():
        if k.startswith(pre) and k[len(pre):].isdigit():
            out[int(k[len(pre):])] = v
    return dict(sorted(out.items()))


def rank_deltas(before: dict, after: dict, name: str) -> dict:
    """{rank: after - before} of a tier counter per rank."""
    return {r: v - before.get(f"spill.{name}.rank{r}", 0) for r, v in rank_values(after, name).items()}


def flatten(prefix: str, obj, out: dict):
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, dict):
            flatten(key + ".", v, out)


def parse_metrics(text: str) -> dict:
    """Prometheus text -> {name without the `tensorfold:` prefix: value}; labelled samples keep their labels."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(" ", 1)
            out[name.split(":", 1)[1] if name.startswith("tensorfold:") else name] = float(value)
        except (ValueError, IndexError):
            continue
    return out


def metrics_counters(m: dict) -> dict:
    """/metrics families mapped onto the /health flat names (prompt_cache_x -> prompt_cache.x; spill_x{rank="N"} ->
    spill.x.rankN, and spill.x over the ranks)."""
    out = {}
    for k, v in m.items():
        labels = {}
        if k.endswith("}") and "{" in k:
            k, _, lab = k[:-1].partition("{")
            for part in lab.split(","):
                n_, _, v_ = part.partition("=")
                labels[n_.strip()] = v_.strip().strip('"')
        if labels:
            if k.startswith("spill_") and set(labels) == {"rank"} and labels["rank"].isdigit():
                put_spill_rank(out, int(labels["rank"]), k[len("spill_"):], v)
            elif k.startswith("kv_shard_part_") and set(labels) == {"part"} and labels["part"].isdigit():
                out[f"kv_shard.part{labels['part']}.{k[len('kv_shard_part_'):]}"] = v
            continue              # other labelled families (pools, histograms) are not counters the checks read
        if k.startswith("prompt_cache_"):
            out["prompt_cache." + k[len("prompt_cache_"):]] = v
        elif k.startswith("spill_"):
            out["spill." + k[len("spill_"):]] = v
        elif k.startswith("kv_shard_"):
            out["kv_shard." + k[len("kv_shard_"):]] = v
        elif k in ("chats_continued", "chats_continued_prefill_tokens"):
            out[k] = v
    aggregate_spill(out)
    return out


def health_counters(h: dict) -> dict:
    """/health's numeric fields flattened (nested ones dotted); prompt_cache.spill per rank as spill.<name>.rank<N>
    (the engine's array of {rank, ...}; a single object, as an older fake gave, counts as rank 0)."""
    out = {}
    if not isinstance(h, dict) or "error" in h:
        return out
    flatten("", h, out)
    sp = (h.get("prompt_cache") or {}).get("spill") if isinstance(h.get("prompt_cache"), dict) else None
    ranks = sp if isinstance(sp, list) else [sp] if isinstance(sp, dict) else []
    for i, o in enumerate(ranks):
        if not isinstance(o, dict):
            continue
        r = o.get("rank", i)
        for k, v in o.items():
            if k != "rank" and isinstance(v, (int, float)) and not isinstance(v, bool):
                put_spill_rank(out, int(r), k, v)
    for k in [k for k in out if k.startswith("prompt_cache.spill.")]:
        del out[k]
    pp = (h.get("kv_shard") or {}).get("per_part") if isinstance(h.get("kv_shard"), dict) else None
    for i, o in enumerate(pp if isinstance(pp, list) else []):
        if isinstance(o, dict):
            p = o.get("part", i)
            for k, v in o.items():
                if k != "part" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"kv_shard.part{p}.{k}"] = v
    aggregate_spill(out)
    return out


class Server:
    def __init__(self, url: str, model: str = DEFAULT_MODEL, timeout: float = 1800, seed: int = 1234):
        url = url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.url, self.model, self.timeout, self.seed = url, model, timeout, seed

    # ------------------------------------------------------------------ plain requests
    def get(self, path: str, timeout: float = 15, raw: bool = False):
        try:
            with OPENER.open(urllib.request.Request(self.url + path, headers=headers()), timeout=timeout) as r:
                data = r.read()
                return data.decode(errors="replace") if raw else json.loads(data)
        except Exception as e:  # noqa: BLE001 - reported, never a traceback
            return None if raw else {"error": f"{type(e).__name__}: {e}"}

    def post_json(self, path: str, body: dict, timeout: float = None):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), headers())
        t0 = time.monotonic()
        try:
            with OPENER.open(req, timeout=timeout or self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}", "http_status": e.code}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        if isinstance(data, dict):
            data["_seconds"] = time.monotonic() - t0
        return data

    def health(self) -> dict:
        return self.get("/health")

    def alive(self) -> bool:
        h = self.health()
        return isinstance(h, dict) and h.get("status") == "ok"

    def wait_healthy(self, timeout: float = 600) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.alive():
                return True
            time.sleep(2)
        return False

    def counters(self) -> dict:
        """Every numeric /health field (nested ones dotted), with /metrics filling what /health lacks."""
        out = health_counters(self.health())
        text = self.get("/metrics", raw=True)
        if text:
            for k, v in metrics_counters(parse_metrics(text)).items():
                out.setdefault(k, v)
        return out

    def tokenize(self, messages: list):
        r = self.post_json("/tokenize", {"model": self.model, "messages": messages,
                                         "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
        return r.get("count"), r.get("max_model_len"), r

    def body(self, messages: list, max_tokens: int, stream: bool = False) -> dict:
        b = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
             "seed": self.seed, "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
        if stream:
            b["stream_options"] = {"include_usage": True}
        return b

    def chat(self, messages: list, max_tokens: int = 64) -> dict:
        """One greedy non-streaming request -> a result row (token_sha strong when the server gives it)."""
        r = self.post_json("/v1/chat/completions", self.body(messages, max_tokens))
        return result_row(messages, r)

    # ------------------------------------------------------------------ streaming
    def stream(self, messages: list, max_tokens: int = 64, cancel_after: float = None,
               cancel_on_first: bool = False, cancel: threading.Event = None, first: threading.Event = None,
               keep_text: bool = False) -> dict:
        """A streamed request. cancel_after closes the connection that many seconds after sending, cancel when that
        event is set (the server sees a client that left); first is set at the first content token. Returns a row with
        ttft, the worst inter-token gap, the gap times, and cancelled=True when it was cut before its end."""
        u = urllib.parse.urlsplit(self.url)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=self.timeout)
        row = {"prompt_sha": sha(json.dumps(messages, sort_keys=True))}
        cut, done = threading.Event(), threading.Event()
        timer = None

        def close():
            cut.set()
            try:
                if conn.sock:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        t0 = time.monotonic()
        text, last, gaps, events, usage, tf, finish, ttft = "", None, [], 0, None, {}, None, None
        try:
            conn.request("POST", (u.path.rstrip("/") or "") + "/v1/chat/completions",
                         json.dumps(self.body(messages, max_tokens, stream=True)).encode(), headers())
            if cancel_after is not None:
                timer = threading.Timer(cancel_after, close)
                timer.start()
            if cancel is not None:
                def watch():
                    while not cancel.wait(0.02):
                        if done.is_set():
                            return
                    close()
                threading.Thread(target=watch, daemon=True).start()
            resp = conn.getresponse()
            if resp.status != 200:
                row["error"] = f"HTTP {resp.status}: {resp.read().decode(errors='replace')[:300]}"
                row["http_status"] = resp.status
                return row
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                events += 1
                if ev.get("error"):
                    row["error"] = str(ev["error"])[:300]
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("tensorfold"):
                    tf.update(ev["tensorfold"])
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.monotonic()
                        if ttft is None:
                            ttft = now - t0
                            if first is not None:
                                first.set()
                        elif last is not None:
                            gaps.append(now - last)
                        last = now
                        text += piece
                        if cancel_on_first:
                            close()
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if cut.is_set():
                    break
        except (OSError, http.client.HTTPException, ValueError) as e:
            if not cut.is_set():
                row["error"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()
            if first is not None:
                first.set()          # never leave a waiter hanging on a stream that failed early
            if timer:
                timer.cancel()
            conn.close()
        row["cancelled"] = cut.is_set() and finish is None
        u2 = usage or {}
        tsha = tf.get("token_sha")
        row.update({"prompt_tokens": u2.get("prompt_tokens"), "completion_tokens": u2.get("completion_tokens"),
                    "cached_tokens": (u2.get("prompt_tokens_details") or {}).get("cached_tokens", tf.get("cached_tokens")),
                    "finish_reason": finish, "token_sha": tsha or sha(text), "token_sha_source": "server" if tsha else "text",
                    "text_sha": sha(text), "text_len": len(text), "events": events,
                    "ttft": round(ttft, 4) if ttft is not None else None,
                    "max_gap": round(max(gaps), 4) if gaps else None, "gaps": len(gaps),
                    "seconds": round(time.monotonic() - t0, 3)})
        if keep_text:
            row["text"] = text
        return row


def result_row(messages, r: dict) -> dict:
    out = {"prompt_sha": sha(json.dumps(messages, sort_keys=True))}
    if "error" in r:
        out["error"] = r["error"]
        if "http_status" in r:
            out["http_status"] = r["http_status"]
        return out
    try:
        ch = r["choices"][0]
        text = ch["message"].get("content") or ""
        u = r.get("usage") or {}
        tf = r.get("tensorfold") or {}
    except (KeyError, IndexError, TypeError) as e:
        out["error"] = f"unexpected reply shape: {e}: {str(r)[:200]}"
        return out
    tsha = tf.get("token_sha")
    out.update({"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "finish_reason": ch.get("finish_reason"), "token_sha": tsha or sha(text),
                "token_sha_source": "server" if tsha else "text", "text_sha": sha(text), "text_len": len(text),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", tf.get("cached_tokens")),
                "prefill_seconds": tf.get("prefill_seconds"), "seconds": round(r.get("_seconds", 0), 3)})
    return out


def same_reply(base: dict, got: dict, allow_weak: bool = False):
    """(ok, detail): got's reply equals the baseline's. Strong (server token_sha on both sides) unless allow_weak."""
    if base is None:
        return False, "no baseline row"
    if "error" in got:
        return False, f"request error: {got['error']}"
    if "error" in base:
        return False, f"baseline row has an error: {base['error']}"
    if base.get("prompt_sha") != got.get("prompt_sha"):
        return False, "prompt differs from the baseline's (different salt or shapes)"
    strong = base.get("token_sha_source") == "server" and got.get("token_sha_source") == "server"
    if strong:
        ok = base["token_sha"] == got["token_sha"]
        return ok, f"token_sha {'equal' if ok else 'DIFFERS'} ({got['token_sha'][:12]} vs baseline {base['token_sha'][:12]})"
    ok = base.get("text_sha") == got.get("text_sha")
    if not allow_weak:
        return False, f"WEAK: no server token_sha on {'both' if base.get('token_sha_source') != 'server' and got.get('token_sha_source') != 'server' else 'one'} side(s); text {'equal' if ok else 'differs'} (use --allow-weak to accept)"
    return ok, f"text_sha {'equal' if ok else 'DIFFERS'} (weak: --allow-weak)"


def delta(before: dict, after: dict, key: str):
    """after - before for a counter, None when the server does not report it."""
    if key not in after:
        return None
    return after[key] - before.get(key, 0)


def deltas(before: dict, after: dict, keys) -> dict:
    return {k: delta(before, after, k) for k in keys}


def wait_until(pred, timeout: float, interval: float = 1.0):
    end = time.monotonic() + timeout
    while True:
        v = pred()
        if v or time.monotonic() >= end:
            return v
        time.sleep(interval)


class Receipt:
    """Per-check results; PASS / FAIL / SKIP (SKIP = not exercised, with the reason). write() saves the JSON."""

    def __init__(self, tool: str, meta: dict):
        self.doc = {"tool": tool, "meta": dict(meta, time=time.strftime("%Y-%m-%dT%H:%M:%S%z")), "checks": []}

    def add(self, name: str, status: str, detail: str = "", **data):
        assert status in ("PASS", "FAIL", "SKIP")
        self.doc["checks"].append(redact({"check": name, "status": status, "detail": detail, **data}))
        detail = redact(detail)
        print(f"{status:4s} {name}" + (f": {detail}" if detail else ""), flush=True)

    def check(self, name: str, ok: bool, detail: str = "", **data):
        self.add(name, "PASS" if ok else "FAIL", detail, **data)
        return ok

    def failed(self):
        return [c["check"] for c in self.doc["checks"] if c["status"] == "FAIL"]

    def finish(self, path: str = None, strict_skip: bool = False) -> int:
        fails = self.failed()
        skips = [c["check"] for c in self.doc["checks"] if c["status"] == "SKIP"]
        bad = fails + (skips if strict_skip else [])
        self.doc["result"] = "PASS" if not bad else "FAIL"
        self.doc["failed"], self.doc["skipped"] = fails, skips
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(redact(self.doc), f, indent=1, sort_keys=True)
        print(f"{self.doc['tool']}: {self.doc['result']} ({len(self.doc['checks'])} checks, {len(fails)} failed"
              f"{', ' + str(len(skips)) + ' skipped' if skips else ''}){(' -> ' + path) if path else ''}", flush=True)
        return 1 if bad else 0


# ---------------------------------------------------------------------- a rank's spill directory
# Operations run as `[shell...] [sudo -n] python3 -c FSOPS <json>`: the same code here and on the worker, and under
# sudo when the engine's containers wrote the files as root. Nothing but these four operations is ever run, and only
# inside the spill root: a path is joined to the resolved root and refused unless it names a regular file inside it
# reached through no symlink (no `..` out, no absolute path), and the file is opened with O_NOFOLLOW (list skips
# anything but regular files). With --sudo that keeps a symlink or a crafted name in the folder from turning a check
# into a root write elsewhere.
FSOPS = r'''
import json, os, stat, sys
a = json.loads(sys.argv[1])
op = a["op"]
root = os.path.realpath(a["root"])
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
def inside(p):
    return p != root and os.path.commonpath([root, p]) == root
def target(rel):
    # the joined path, resolved: refused unless it names a regular file inside the root, reached through no symlink
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        raise SystemExit("refused: not a path inside the spill root: %r" % (rel,))
    p = os.path.normpath(os.path.join(root, rel))
    parent = os.path.realpath(os.path.dirname(p))
    if not inside(p) or (parent != root and not inside(parent)) or os.path.join(parent, os.path.basename(p)) != p:
        raise SystemExit("refused: %r resolves outside the spill root %s or through a symlink" % (rel, root))
    st = os.lstat(p)
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit("refused: %r is not a regular file" % (rel,))
    return p
def opened(p, flags):
    fd = os.open(p, flags | NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit("refused: %s is not a regular file" % p)
    return fd
def files():
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                out[os.path.relpath(p, root)] = [st.st_size, st.st_mtime]
    return out
if op == "list":
    r = files()
elif op == "corrupt":
    fd = opened(target(a["path"]), os.O_RDWR)
    try:
        n = os.fstat(fd).st_size
        at = min(max(0, a.get("at", n // 2)), max(0, n - 1))
        b = os.pread(fd, 1, at)
        os.pwrite(fd, bytes([(b[0] if b else 0) ^ 0xFF]), at)
        os.fsync(fd)
    finally:
        os.close(fd)
    r = {"path": a["path"], "size": n, "at": at}
elif op == "truncate":
    fd = opened(target(a["path"]), os.O_WRONLY)
    try:
        n = os.fstat(fd).st_size
        to = a.get("to", n // 2)
        os.ftruncate(fd, to)
    finally:
        os.close(fd)
    r = {"path": a["path"], "size": n, "to": to}
elif op == "remove":
    p = target(a["path"])
    dfd = os.open(os.path.dirname(p), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | NOFOLLOW)
    try:
        os.unlink(os.path.basename(p), dir_fd=dfd)
    finally:
        os.close(dfd)
    r = {"path": a["path"]}
elif op == "statvfs":
    s = os.statvfs(root)
    r = {"free": s.f_bavail * s.f_frsize, "total": s.f_blocks * s.f_frsize}
else:
    raise SystemExit("unknown op " + op)
print(json.dumps(r))
'''


class RankDir:
    """One rank's spill root (the host path the engine's --spill-dir maps to), reached here or over a shell prefix."""

    def __init__(self, root: str, shell: str = "", sudo: bool = False, rank: int = 0):
        self.root, self.shell, self.sudo, self.rank = root, shell.split() if shell else [], sudo, rank

    def run(self, op: str, **kw):
        arg = json.dumps(dict(kw, op=op, root=self.root))
        cmd = list(self.shell)
        if self.sudo:
            cmd += ["sudo", "-n"]
        if self.shell:      # a remote shell joins its arguments into one command line: quote them
            import shlex
            cmd += ["python3", "-c", shlex.quote(FSOPS), shlex.quote(arg)]
        else:
            cmd += [sys.executable or "python3", "-c", FSOPS, arg]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(f"rank {self.rank} {op} failed ({p.returncode}): {p.stderr.strip()[-300:]}")
        return json.loads(p.stdout)

    def list(self) -> dict:
        return self.run("list")

    def new_files(self, before: dict) -> dict:
        """Names that did not exist in `before` (a later touch of an older manifest changes only its mtime)."""
        return {k: v for k, v in self.list().items() if k not in before}


# A recipe's own mark at a spill folder's root (the DeepSeek recipe's start.sh writes one once the folder's checks pass;
# this recipe's scripts/port.sh writes none yet, open issue 4 in docs/glm-port/recipe.md): not the store's, and never a
# temporary.
SPILL_MARKER = ".glm53-tensorfold-spill"


def kind_of(path: str) -> str:
    """spill-store's layout: blocks/<name>.blk, points/<key>.{ids,st,man}; its temporaries 'temp'; the recipe's root
    marker (SPILL_MARKER, exactly that name at the root) 'marker'; anything else is 'other'."""
    if path == SPILL_MARKER:
        return "marker"
    for ext, kind in ((".blk", "block"), (".st", "small"), (".man", "manifest"), (".ids", "ids")):
        if path.endswith(ext):
            return kind
    return "temp" if ".tmp" in path or path.rsplit("/", 1)[-1].startswith(".") else "other"
