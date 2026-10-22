#!/usr/bin/env python3
"""Bit-exactness harness: does a NEW engine image give the same greedy outputs as the OLD one?

  python3 -I exact.py record  --out old.json --label old-img [--drafts auto|on|off] [--url URL]
  python3 -I exact.py record  --out new.json --label new-img --like old.json     # same salt + prompt shapes
  python3 -I exact.py compare old.json new.json                                   # exit 1 on any mismatch

record sends a fixed prompt set (greedy: temperature 0 + seed, thinking off, non-streaming) and stores per prompt:
prompt/completion tokens, finish reason, tensorfold.token_sha (sha256 of the reply token ids; falls back to sha256 of the
exact text and says so), text sha256, cached tokens. It also runs intra-run checks (repeat == first, resume suffix replayed,
8 concurrent == the same 8 solo) and exits 1 if one fails or a request errored.
Prompt set 2 (meta.prompt_set; a set-1 record cannot be used with --like). Cases: len_1, len_100, len_2048, len_2049,
len_4096, len_8000, len_20000, len_30000, len_50000, len_64000, len_100000 (cold: own salt per case; >=30000 sized from
a fit of the 8000/20000 calibrations, check prompt_tokens), conv_A1 B1 C1 A2 B2 C2 A3 (C1 scenario: 3 conversations with distinct
~6K-token system prompts, interleaved A1 B1 C1 A2 B2 C2 A3, each turn appends the previous reply; cached_tokens is
shown side by side in compare as information only), repeat_2048
(identical prompt again, cached), resume_1..3 (one long prefix, 3 short suffixes, sequential) + resume_1_again,
conc_1..8 (8 at once, the qualified eight-stream baseline; --streams 4 for the old four) + solo_1..8. Prompt sizes are calibrated against usage.prompt_tokens (not exact for the long ones:
compare the recorded prompt_tokens). --like FILE reuses the salt and word counts of an earlier record so the prompts
are byte-identical. Without --like the salt is random (so a cold run is cold even on a warm server).
Compatibility: a record made before the eight-stream cases (meta.conc_streams absent: 4) still serves as --like: the
shapes of conc_5..8 are then derived from its conc_1..4 shapes (deterministically, so two records --like the same file
match), and compare treats conc_k/solo_k beyond a record's own conc_streams as absent by design, not as a mismatch.
Window: cold cases past the window (--context, CONTEXT, 131,072 by default; checklib.fit) are left out and listed in
meta.over_window, and compare notes a case one record left out that way instead of calling it a mismatch (at the
default every case fits). It needs the Sparks to record (the image before and after a change); compare runs anywhere,
and tools/selftest.py runs both here against tools/fake_server.py. From the DeepSeek recipe (MiaAI-Lab's exact.py, with
the eight-stream cases of Robert Wojciechowski (GitHub @wojo, X @rwojo)), for GLM-5.3-Flash on the Zig engine.
Stdlib only. API key: checklib's rule (API_KEY, else TENSORFOLD_API_KEY, else --api-key-file). Proxies are never used.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import secrets
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
import checklib as C  # noqa: E402

SCHEMA = 1
PROMPT_SET = 2   # 2 adds len_30000/50000/64000/100000 and the conv_* scenario; records of set 1 cannot be used with --like
DEFAULT_MODEL = C.DEFAULT_MODEL
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
COLD_SIZES = [("len_1", 1), ("len_100", 100), ("len_2048", 2048), ("len_2049", 2049), ("len_4096", 4096),
              ("len_8000", 8000), ("len_20000", 20000), ("len_30000", 30000), ("len_50000", 50000),
              ("len_64000", 64000), ("len_100000", 100000)]
PREDICT_FROM = 30000      # targets at or above this are predicted from the 8000/20000 fits, not calibrated (too slow)
CONV_SYS_TARGET = 6000
CONV_ORDER = ["A1", "B1", "C1", "A2", "B2", "C2", "A3"]
CONV_Q = {"A": ["What is the first topic in the text above?", "Name two words that repeat in the text.", "Say one short sentence about the weather."],
          "B": ["Summarize the text above in five words.", "Which word comes last in the text?", "Reply with a number between one and nine."],
          "C": ["Give a title for the text above.", "List three nouns from the text.", "Say hello in a different way."]}
RESUME_TARGET = 8000
CONC_TARGETS = [600, 1200, 1800, 2400, 3000, 3600, 4200, 4800]   # the first four are the original cases
OLD_STREAMS = 4                                                     # records without meta.conc_streams
SUFFIXES = ["Summarize the text above in one sentence.",
            "List the three most frequent words in the text above.",
            "Write one question a reader could ask about the text above."]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


headers = C.headers


def sha(s) -> str:
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


_units_cache = {}


def words(n: int, seed: int) -> str:
    """The first n words of a deterministic prose stream (prefix-stable: words(n) is a prefix of words(n+1))."""
    if seed not in _units_cache:
        _units_cache[seed] = (random.Random(seed), [])
    rng, units = _units_cache[seed]
    while len(units) < n:
        k = rng.randint(6, 16)
        s = [rng.choice(WORDS) for _ in range(k)]
        s[-1] += "."
        units.extend(s)
    return " ".join(units[:n])


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers())
    t0 = time.monotonic()
    try:
        with OPENER.open(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except Exception as e:  # noqa: BLE001 - report, do not crash
        return {"error": f"{type(e).__name__}: {e}"}
    data["_seconds"] = time.monotonic() - t0
    return data


def get_json(url, path, timeout=10):
    try:
        with OPENER.open(urllib.request.Request(url + path, headers=headers()), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


class Runner:
    def __init__(self, a):
        self.a = a

    def ask(self, content, max_tokens=None) -> dict:
        """content: a user string, or a full messages list."""
        messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
        body = {"model": self.a.model, "messages": messages,
                "max_tokens": max_tokens or self.a.max_tokens, "temperature": 0, "seed": self.a.seed,
                "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        r = post(self.a.url, body, self.a.timeout)
        out = {"prompt_sha": sha(json.dumps(messages, sort_keys=True))}
        if "error" in r:
            out["error"] = r["error"]
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
        out.update({
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "token_sha": tsha if tsha else sha(text), "token_sha_source": "server" if tsha else "text",
            "text_sha": sha(text), "text_len": len(text),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", tf.get("cached_tokens")),
            "drafts": tf.get("drafts"), "sampling": tf.get("sampling"), "seconds": round(r["_seconds"], 3)})
        out["_text"] = text
        if self.a.keep_text:
            out["text"] = text
        return out

    def calibrate(self, target: int, tag: str, tol: int):
        """Word count n whose whole prompt is `target` tokens (within tol), found with max_tokens=1 on a throwaway salt."""
        if target <= 1:
            return 0, 1
        n = max(1, int(target / 1.2))
        best = (10 ** 9, n, None)
        for _ in range(8):
            r = self.ask(self.text(f"cal-{secrets.token_hex(4)}-{tag}", n, SUFFIXES[0]), max_tokens=1)
            pt = r.get("prompt_tokens")
            if pt is None:
                break
            diff = pt - target
            if abs(diff) < best[0]:
                best = (abs(diff), n, pt)
            if abs(diff) <= tol:
                return n, pt
            step = max(1, round(abs(diff) / 1.15))
            n = max(1, n - step if diff > 0 else n + step)
        log(f"calibrate {tag}: target {target} not hit exactly; using {best[1]} words (off by {best[0]})")
        return best[1], best[2]

    def text(self, salt, n, suffix):
        return f"[ref {salt}]\n" + words(n, self.a.seed) + ("\n\n" + suffix if suffix else "")


def log(*m):
    print(*m, file=sys.stderr, flush=True)


def record(a) -> int:
    like = json.load(open(a.like)) if a.like else None
    if like and like["meta"].get("prompt_set", 1) != PROMPT_SET:
        log(f"--like {a.like} is prompt set {like['meta'].get('prompt_set', 1)}, this exact.py is set {PROMPT_SET}: record the OLD image again with this version")
        return 2
    salt = like["meta"]["salt"] if like else secrets.token_hex(6)
    shape = dict(like["meta"]["shape"]) if like else {}
    if like and like["meta"].get("seed") != a.seed:
        log(f"warning: --seed {a.seed} differs from the --like file's {like['meta'].get('seed')}; the prompts will differ")
    run = Runner(a)
    health = get_json(a.url, "/health")
    log(f"record: label={a.label!r} url={a.url} salt={salt} like={a.like}")
    limit = C.fit(a.context, a.max_tokens)
    cold = [(n, t) for n, t in COLD_SIZES if t <= limit]
    over = [n for n, t in COLD_SIZES if t > limit]
    if over:
        log(f"  left out (past the window, {a.context} tokens: at most {limit} a prompt): {over}")
    todo = [(n, t, 0 if t <= 2049 else max(2, t // 100)) for n, t in cold] + [("resume", RESUME_TARGET, 80)]
    streams = a.streams
    todo += [(f"conc_{i+1}", t, 20) for i, t in enumerate(CONC_TARGETS[:streams])] + [("convsys", CONV_SYS_TARGET, 60)]
    cal = {}
    for name, tgt, tol in todo:
        if name in shape:
            continue
        if like and name.startswith("conc_") and int(name[5:]) > like["meta"].get("conc_streams", OLD_STREAMS):
            # an older record: derive the extra streams' shapes from its own conc shapes (deterministic)
            known = [(shape[f"conc_{i+1}"], CONC_TARGETS[i]) for i in range(OLD_STREAMS) if f"conc_{i+1}" in shape]
            ratio = sum(w for w, _ in known) / sum(t for _, t in known)
            shape[name] = max(1, int(tgt * ratio))
            log(f"  shape {name}: {shape[name]} words (derived from the --like record's conc shapes)")
            continue
        if like:
            log(f"--like file has no shape for {name}")
            return 2
        if a.no_calibrate:
            shape[name] = max(0, int(tgt / 1.15)) if tgt > 1 else 0
        elif tgt >= PREDICT_FROM and "len_8000" in cal and "len_20000" in cal:
            (n1, p1), (n2, p2) = cal["len_8000"], cal["len_20000"]
            b = (p2 - p1) / (n2 - n1)
            shape[name] = max(1, int((tgt - (p1 - b * n1)) / b))
        else:
            shape[name], pt = run.calibrate(tgt, name, tol if name != "len_100" else 1)
            cal[name] = (shape[name], pt)
        log(f"  shape {name}: {shape[name]} words (target {tgt} tokens)")
    if shape["len_2048"] == shape["len_2049"]:
        log("warning: len_2048 and len_2049 got the same word count (not distinct prompts)")
    cases, checks = {}, []

    def put(name, target, content):
        r = run.ask(content)
        r.pop("_text", None)
        r["target_tokens"] = target
        cases[name] = r
        log(f"  {name}: " + (r["error"] if "error" in r else
            f"prompt={r['prompt_tokens']} out={r['completion_tokens']} {r['finish_reason']} cached={r['cached_tokens']} "
            f"sha={r['token_sha'][:12]}({r['token_sha_source']}) {r['seconds']}s"))
        return r

    for name, tgt in cold:
        n = shape[name]
        content = "Hi" if tgt == 1 else run.text(f"{salt}-{name}", n, "Summarize the text above in one sentence.")
        put(name, tgt, content)
    put("repeat_2048", 2048, run.text(f"{salt}-len_2048", shape["len_2048"], "Summarize the text above in one sentence."))
    n = shape["resume"]
    for i, s in enumerate(SUFFIXES):
        put(f"resume_{i+1}", RESUME_TARGET, run.text(f"{salt}-resume", n, s))
    put("resume_1_again", RESUME_TARGET, run.text(f"{salt}-resume", n, SUFFIXES[0]))
    conc = [(f"conc_{i+1}", run.text(f"{salt}-conc{i+1}", shape[f"conc_{i+1}"], "Summarize the text above in one sentence."))
            for i in range(streams)]
    with concurrent.futures.ThreadPoolExecutor(streams) as ex:
        futs = {nm: ex.submit(run.ask, c) for nm, c in conc}
        for (nm, _), tgt in zip(conc, CONC_TARGETS):
            r = futs[nm].result()
            r.pop("_text", None)
            r["target_tokens"] = tgt
            cases[nm] = r
            log(f"  {nm} ({streams} at once): " + (r["error"] if "error" in r else
                f"prompt={r['prompt_tokens']} out={r['completion_tokens']} sha={r['token_sha'][:12]} {r['seconds']}s"))
    for i, (nm, c) in enumerate(conc):
        put(f"solo_{i+1}", CONC_TARGETS[i], c)

    # C1 scenario: 3 conversations with distinct system prompts, interleaved; each turn appends the previous reply
    convs = {k: [{"role": "system", "content": f"[ref {salt}-conv{k}]\n" + words(shape["convsys"], a.seed + ord(k))}] for k in "ABC"}
    dead = set()
    for turn in CONV_ORDER:
        k, t = turn[0], int(turn[1]) - 1
        if k in dead:
            continue
        convs[k].append({"role": "user", "content": CONV_Q[k][t]})
        r = run.ask(list(convs[k]))
        text = r.pop("_text", "")
        r["target_tokens"] = CONV_SYS_TARGET
        cases[f"conv_{turn}"] = r
        log(f"  conv_{turn}: " + (r["error"] if "error" in r else
            f"prompt={r['prompt_tokens']} out={r['completion_tokens']} cached={r['cached_tokens']} sha={r['token_sha'][:12]} {r['seconds']}s"))
        if "error" in r:
            dead.add(k)
        else:
            convs[k].append({"role": "assistant", "content": text})

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"  check {name}: {'ok' if ok else 'FAIL'} {detail}")

    def same(x, y):
        return "error" not in cases[x] and "error" not in cases[y] and cases[x]["token_sha"] == cases[y]["token_sha"]
    chk("repeat_2048 == len_2048", same("repeat_2048", "len_2048"))
    chk("resume_1_again == resume_1", same("resume_1_again", "resume_1"))
    for i in range(1, streams + 1):
        chk(f"solo_{i} == conc_{i}", same(f"solo_{i}", f"conc_{i}"))
    c = cases["repeat_2048"].get("cached_tokens")
    log(f"  info: repeat_2048 cached_tokens={c} (informational: a 2048-token prompt may keep no snapshot)")
    for nm in ("resume_2", "resume_3"):
        c = cases[nm].get("cached_tokens")
        if c is None:
            checks.append({"check": f"{nm} cache hit", "ok": True, "detail": "server reports no cached_tokens (not checked)"})
        else:
            chk(f"{nm} cache hit (cached_tokens>0)", c > 0, f"cached={c}")
    drafts_seen = sorted({str(c.get("drafts")) for c in cases.values() if "error" not in c})
    weak = sorted(n for n, c in cases.items() if "error" not in c and c.get("token_sha_source") != "server")
    if weak:
        log(f"WARNING: the server gave no tensorfold.token_sha for {len(weak)} case(s): this record is WEAK (text hash only); compare refuses it without --allow-weak")
    meta = {"schema": SCHEMA, "prompt_set": PROMPT_SET, "conc_streams": streams, "context": a.context, "over_window": over, "weak": bool(weak), "label": a.label, "drafts_label": a.drafts, "drafts_observed": drafts_seen,
            "url": a.url, "model": a.model, "salt": salt, "seed": a.seed, "max_tokens": a.max_tokens, "shape": shape,
            "like": a.like, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "health": {k: health.get(k) for k in ("status", "model", "max_batch_size", "error")}}
    errors = [n for n, c in cases.items() if "error" in c]
    doc = {"meta": meta, "cases": cases, "checks": checks}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    bad = [c["check"] for c in checks if not c["ok"]]
    log(f"wrote {a.out}: {len(cases)} cases, {len(errors)} errors {errors}, {len(bad)} failed checks {bad}")
    return 1 if errors or bad else 0


def compare(pa, pb, allow_weak=False) -> int:
    A, B = json.load(open(pa)), json.load(open(pb))
    bad = 0
    ma, mb = A["meta"], B["meta"]
    print(f"A: {pa} label={ma.get('label')!r} drafts={ma.get('drafts_label')}/{ma.get('drafts_observed')} {ma.get('time')}")
    print(f"B: {pb} label={mb.get('label')!r} drafts={mb.get('drafts_label')}/{mb.get('drafts_observed')} {mb.get('time')}")
    for k in ("prompt_set", "salt", "seed", "max_tokens", "model"):
        if ma.get(k) != mb.get(k):
            print(f"MISMATCH meta.{k}: {ma.get(k)!r} vs {mb.get(k)!r} (prompts differ; use record --like)")
            bad += 1
    if ma.get("drafts_observed") != mb.get("drafts_observed"):
        print(f"note: drafts differ ({ma.get('drafts_observed')} vs {mb.get('drafts_observed')}): drafted must equal undrafted")
    for who, d in (("A", A), ("B", B)):
        if d["meta"].get("weak") or any(c.get("token_sha_source") == "text" for c in d["cases"].values() if "error" not in c):
            if allow_weak:
                print(f"note: {who} is a WEAK record (no server token_sha); only text hashes were compared (--allow-weak)")
            else:
                print(f"MISMATCH {who} is a WEAK record (no tensorfold.token_sha from the server); refusing to call it identical (use --allow-weak to accept text hashes)")
                bad += 1
    for who, d, p in (("A", A, pa), ("B", B, pb)):
        for c in d.get("checks", []):
            if not c["ok"]:
                print(f"MISMATCH {who} intra-run check failed: {c['check']} {c['detail']}")
                bad += 1
    names = sorted(set(A["cases"]) | set(B["cases"]), key=lambda n: list(A["cases"]).index(n) if n in A["cases"] else 999)
    for n in names:
        a, b = A["cases"].get(n), B["cases"].get(n)
        if a is None or b is None:
            lacking = ma if a is None else mb
            if n in (lacking.get("over_window") or []):
                print(f"note     {n}: only in {'B' if a is None else 'A'} (past {'A' if a is None else 'B'}'s window, "
                      f"{lacking.get('context')} tokens)")
                continue
            k = n.split("_", 1)[1] if n.startswith(("conc_", "solo_")) else ""
            if k.isdigit() and int(k) > lacking.get("conc_streams", OLD_STREAMS):
                print(f"note     {n}: only in {'B' if a is None else 'A'} ({'A' if a is None else 'B'} recorded "
                      f"{lacking.get('conc_streams', OLD_STREAMS)} concurrent streams)")
                continue
            print(f"MISMATCH {n}: present only in {'B' if a is None else 'A'}")
            bad += 1
            continue
        if "error" in a or "error" in b:
            print(f"MISMATCH {n}: request error A={a.get('error')} B={b.get('error')}")
            bad += 1
            continue
        diffs = []
        for k in ("prompt_sha", "prompt_tokens", "completion_tokens", "finish_reason", "text_sha"):
            if a.get(k) != b.get(k):
                diffs.append(f"{k}: {a.get(k)} vs {b.get(k)}")
        if a.get("token_sha_source") == b.get("token_sha_source"):
            if a["token_sha"] != b["token_sha"]:
                diffs.append(f"token_sha({a['token_sha_source']}): {a['token_sha'][:16]} vs {b['token_sha'][:16]}")
        else:
            print(f"note {n}: token_sha sources differ ({a.get('token_sha_source')} vs {b.get('token_sha_source')}); text_sha decides")
        if diffs:
            bad += 1
            print(f"MISMATCH {n}: " + "; ".join(diffs) + f"; cached(A/B)={a.get('cached_tokens')}/{b.get('cached_tokens')}")
        else:
            print(f"ok       {n}: prompt={a['prompt_tokens']} out={a['completion_tokens']} sha={a['token_sha'][:12]} cached(A/B)={a.get('cached_tokens')}/{b.get('cached_tokens')}")
    print(f"RESULT: {'IDENTICAL' if bad == 0 else 'DIFFERENT'} ({bad} problem(s), {len(names)} cases)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")),
                   help="server root or .../v1 (default API_URL or http://127.0.0.1:$PORT)")
    r.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    r.add_argument("--context", type=int, default=C.default_context(),
                   help="the server's window (default CONTEXT, else 131072): cold cases past it are left out")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--drafts", choices=("auto", "on", "off"), default="auto", help="label only: how the server was started")
    r.add_argument("--like", help="earlier record: reuse its salt and prompt shapes (required for a later compare)")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--max-tokens", type=int, default=192)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--no-calibrate", action="store_true")
    r.add_argument("--streams", type=int, choices=(4, 8), default=8,
                   help="concurrent cases: 8 (default, the eight-stream baseline) or 4 (the original set)")
    r.add_argument("--keep-text", action="store_true", help="store reply text too (debugging)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--allow-weak", action="store_true", help="accept records without the server's token_sha (text hashes only)")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    if a.mode == "compare":
        return compare(a.a, a.b, a.allow_weak)
    a.url = a.url.rstrip("/")
    if a.url.endswith("/v1"):
        a.url = a.url[:-3]
    return record(a)


if __name__ == "__main__":
    sys.exit(main())
