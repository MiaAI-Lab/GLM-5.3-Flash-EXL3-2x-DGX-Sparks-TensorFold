"""CPU-only check that a phone photo (an MPO JPEG: the picture plus an auxiliary frame) is read (patch 0110, issue #201).

  python3 -B tools/test_mpo_first_frame.py --source-root /path/to/patched/src
Use --expect-stock on the source without 0110 to see the 400 the issue reports. Needs Pillow; no torch, GPU or sockets.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from PIL import Image
    from tensorfold.vision.images import ImageSource, load_images
    from tensorfold.vision.images_http import ImageInputError

    def url(data: bytes, media: str) -> str:
        return f"data:{media};base64,{base64.b64encode(data).decode()}"

    def save(fmt: str, frames: list, **kw) -> bytes:
        buf = io.BytesIO()
        frames[0].save(buf, format=fmt, save_all=len(frames) > 1, append_images=frames[1:], **kw)
        return buf.getvalue()

    def load(data: bytes, media: str):
        try:
            return load_images([ImageSource(url(data, media))]), None
        except ImageInputError as exc:
            return None, str(exc)

    picture = Image.new("RGB", (64, 48), (200, 30, 30))
    auxiliary = Image.new("RGB", (16, 12), (30, 30, 200))
    mpo = save("MPO", [picture, auxiliary])
    assert Image.open(io.BytesIO(mpo)).n_frames == 2 and Image.open(io.BytesIO(mpo)).format == "MPO", "not an MPO"
    got, error = load(mpo, "image/jpeg")
    if args.expect_stock:
        assert error and "multipage" in error, f"stock source accepted an MPO ({error})"
        print(f"stock: the MPO is refused, as reported ({error})")
        return
    assert error is None, error
    image = got[0]
    assert (image.width, image.height) == (64, 48), (image.width, image.height)
    r, g, b = image.pixels[:3]
    assert r > 150 and g < 90 and b < 90, ("first frame was not the picture", (r, g, b))
    # an auxiliary frame far larger than the limits is never decoded: the picture's own size is what is checked
    big = save("MPO", [picture, Image.new("RGB", (8192, 8192), (0, 0, 0))])
    assert load(big, "image/jpeg")[1] is None, "an MPO with a large auxiliary frame was refused"
    # a plain JPEG and a PNG are unchanged; other multi-frame images are still refused
    assert load(save("JPEG", [picture]), "image/jpeg")[1] is None
    assert load(save("PNG", [picture]), "image/png")[1] is None
    for fmt, media in (("WEBP", "image/webp"), ("PNG", "image/png")):
        frames = [picture, Image.new("RGB", (64, 48), (0, 200, 0))]
        kw = {"duration": 100, "loop": 0} if fmt == "WEBP" else {}
        error = load(save(fmt, frames, **kw), media)[1]
        assert error and "multipage" in error, (fmt, error)
    print("MPO: the first frame is read, auxiliary frames are not decoded, animated WebP and APNG are still refused")


if __name__ == "__main__":
    main()
