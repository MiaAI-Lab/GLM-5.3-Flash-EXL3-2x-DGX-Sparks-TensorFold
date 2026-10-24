"""CPU-only test that spill_check's record leaves every item's hit able to resume its seed's kept point.

  python3 -I tools/test_spill_items.py

An item's seed is `doc + Q_SEED`, its hit `doc + Q_HIT`; they diverge at the question. The GLM engine keeps a prompt's
state at its last 64-token boundary (`at <= len - 1`), so when the seed's length lands a few tokens past a boundary
the kept point holds part of Q_SEED and the hit can never resume it, from memory or from disk (g2 and g3's D1: seed
37,963 tokens kept at 37,952, the hit 37,964 resumed 0 in the cache-on baseline record already, so dual-restore read
only D2 from disk and prefilled D1 from 0). Against tools/fake_server.py with that rule (--grid 64 --window 1), a
dual item sized to land there must still come out of `record` with a hit that resumed its seed, and the baseline must
say what it padded so `run` sends the same prompts. Exit 1 on a failure.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checklib as C  # noqa: E402
import spill_check as S  # noqa: E402

PY = sys.executable or "python3"
GRID = 64
FAST = ["--prefill-us-per-token", "1", "--decode-ms-per-token", "0"]


def fake_len(tokens, q):
    """The fake's prompt length for an item of `tokens` asking `q` (unpadded): <user> [ref salt-name] words q <assistant>."""
    return 1 + 2 + C.words_for(tokens) + len(q.split()) + 1


def bad(tokens):
    """Whether the fake keeps the seed past where seed and hit diverge (the last boundary covers part of Q_SEED)."""
    n = fake_len(tokens, S.Q_SEED)
    kept = (n - 1) // GRID * GRID
    return kept > n - len(S.Q_SEED.split()) - 1


def main():
    dual = next(t for t in range(30000, 31000) if bad(t))
    small = next(t for t in range(3000, 4000) if not bad(t))
    fails = []
    with tempfile.TemporaryDirectory() as work:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        url = f"http://127.0.0.1:{port}"
        log = open(os.path.join(work, "fake.log"), "w")
        p = subprocess.Popen([PY, "-I", os.path.join(HERE, "fake_server.py"), "--port", str(port), "--grid", str(GRID),
                              "--window", "1", "--mem-points", "64"] + FAST, stdout=log, stderr=subprocess.STDOUT)
        try:
            end = time.monotonic() + 20
            while True:
                try:
                    urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/health", timeout=2).read()
                    break
                except OSError:
                    if time.monotonic() > end:
                        raise
                    time.sleep(0.1)
            out = os.path.join(work, "base.json")
            r = subprocess.run([PY, "-I", os.path.join(HERE, "spill_check.py"), "record", "--url", url, "--out", out,
                                "--sizes", "", "--dual-tokens", str(dual), "--cancel-tokens", str(small),
                                "--kill-tokens", str(small), "--floor-tokens", str(small)],
                               capture_output=True, text=True, timeout=600)
            print(r.stdout + r.stderr, end="")
            if r.returncode != 0:
                fails.append(f"record exited {r.returncode}")
            if os.path.exists(out):
                base = json.load(open(out))
                for name, row in sorted(base["rows"].items()):
                    if name == S.DECODER:
                        continue
                    # the fake keeps the seed at its last boundary: the hit must resume exactly there
                    kept = (row["seed"]["prompt_tokens"] - 1) // GRID * GRID
                    if row["hit"].get("cached_tokens") != kept:
                        fails.append(f"{name}: hit resumed {row['hit'].get('cached_tokens')}, not its seed's kept "
                                     f"point {kept} ({row['hit'].get('prompt_tokens')} tokens)")
                # run builds the same prompts the record sent: the hit it would send now resumes in this server too
                srv = C.Server(url, base["meta"]["model"], 60, base["meta"]["seed"])
                pads = base["meta"].get("pads", {})
                for name in ("D1", "D2"):
                    try:
                        m = S.msgs(base["meta"]["salt"], name, base["meta"]["items"][name], base["meta"]["seed"],
                                   S.Q_HIT, pads.get(name, 0))
                    except TypeError as e:
                        fails.append(f"spill_check.msgs takes no pad ({e})")
                        break
                    got = srv.chat(m, base["meta"]["max_tokens"])
                    if got.get("token_sha") != base["rows"][name]["hit"].get("token_sha"):
                        fails.append(f"{name}: run's hit prompt differs from the record's ({got.get('prompt_tokens')} "
                                     f"vs {base['rows'][name]['hit'].get('prompt_tokens')} tokens)")
            else:
                fails.append("no baseline written")
        finally:
            p.terminate()
            p.wait(10)
    print(f"dual items of {dual} tokens (a kept point past the question), others {small}")
    print("FAIL: " + "; ".join(fails) if fails else "PASS: every item's hit resumes its seed's kept point")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
