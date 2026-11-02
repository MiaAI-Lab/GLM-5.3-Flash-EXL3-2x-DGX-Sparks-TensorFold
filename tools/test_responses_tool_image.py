"""CPU-only check of image parts inside a Responses ``function_call_output`` (patch 0109, issue #162).

  python3 -B tools/test_responses_tool_image.py --source-root /path/to/patched/src
Use --expect-stock on the source without 0109 to see the 400 the issue reports. Needs no torch, GPU or sockets.
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

    url = "data:image/png;base64,iVBORw0KGgo="
    call = {"type": "function_call", "call_id": "c1", "name": "screenshot", "arguments": "{}"}

    def run(output):
        body = {"model": "x", "store": False, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Look."}]}, call,
            {"type": "function_call_output", "call_id": "c1", "output": output}]}
        try:
            return translate(body, Store()).chat["messages"], None
        except RequestError as exc:
            return None, str(exc)

    shot = [{"type": "input_text", "text": "done"}, {"type": "input_image", "image_url": url}]
    messages, error = run(shot)
    if args.expect_stock:
        assert error, "stock source accepted an image in a function_call_output"
        print(f"stock: refused, as reported ({error})")
        return
    assert error is None, error
    tool = [m for m in messages if m["role"] == "tool"]
    assert len(tool) == 1 and isinstance(tool[0]["content"], list), tool
    kinds = [p["type"] for p in tool[0]["content"]]
    assert kinds == ["text", "image_url"], kinds
    assert tool[0]["content"][1]["image_url"]["url"] == url, tool[0]
    # text-only outputs keep their old shape: a string for a string, the joined text for input_text parts
    for output, expect in (("ok", "ok"), ([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}], "ab")):
        got, error = run(output)
        assert error is None and [m for m in got if m["role"] == "tool"][0]["content"] == expect, (output, got, error)
    # anything else is still refused
    for bad in ([{"type": "input_file", "file_id": "f"}], [{"type": "input_text", "text": "a"}, 5], 7, None):
        assert run(bad)[1], f"{bad!r} was accepted"
    print("function_call_output: image parts reach the tool message, text outputs unchanged, other parts refused")


if __name__ == "__main__":
    main()
