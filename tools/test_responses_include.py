"""CPU-only check of the Responses translator's ``include`` handling (issue #73).

  python3 -B tools/test_responses_include.py --source-root /path/to/patched/src
Use --expect-stock on the unpatched source to see the 400 the issue reports. Needs no torch, GPU or sockets.
"""
from __future__ import annotations

import argparse
import sys


class Store:
    def conversation(self, _id):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from tensorfold.server.errors import RequestError
    from tensorfold.server.responses_translate import translate

    body = {"model": "x", "store": False, "input": [{"role": "user", "content": [{"type": "input_text", "text": "Say OK."}]}]}

    def run(include):
        try:
            translate({**body, "include": include}, Store())
            return None
        except RequestError as exc:
            return str(exc)

    if args.expect_stock:
        assert run(["reasoning.encrypted_content"]), "stock source accepted include"
        print("stock: include refused, as reported")
        return
    for ok in ([], None, ["reasoning.encrypted_content"], ["reasoning.encrypted_content", "message.output_text.logprobs"]):
        assert run(ok) is None, (ok, run(ok))
    for bad in (["nope"], "reasoning.encrypted_content", [1], {"a": 1}):
        assert run(bad), f"{bad!r} was accepted"
    assert translate({**body, "include": ["reasoning.encrypted_content"]}, Store()).chat["messages"] == \
        translate(body, Store()).chat["messages"], "include changed the chat request"
    print("include: known values accepted and ignored, unknown or malformed ones refused")


if __name__ == "__main__":
    main()
