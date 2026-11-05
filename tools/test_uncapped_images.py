"""CPU checks: python3 tools/test_uncapped_images.py --source-root PATCHED_SOURCE.

Needs numpy and Pillow; no model, GPU or server. Exercises the actual vision
input functions, including aggregate budgets on the cached-picture path.
"""
import argparse
import base64
import hashlib
import io
import os
import sys
import unittest
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

p = argparse.ArgumentParser()
p.add_argument('--source-root', required=True)
a, rest = p.parse_known_args()
sys.path.insert(0, a.source_root)
from PIL import Image
from tensorfold.vision import glm, images
from tensorfold.vision.images_http import ImageInputError

buf = io.BytesIO()
Image.new('RGB', (8, 8), 'red').save(buf, format='PNG')
DATA = buf.getvalue()
URL = 'data:image/png;base64,' + base64.b64encode(DATA).decode()
SOURCE = images.ImageSource(URL)


def frontend(limits):
    # Input validation does not require the tower or tokenizer.
    obj = object.__new__(glm.GlmVision)
    obj.limits = limits
    obj.allow_urls = False
    obj._pictures = OrderedDict()
    return obj


class UncappedImages(unittest.TestCase):
    def test_default_and_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            default = glm.GlmVisionLimits.from_env()
        self.assertEqual(default.max_images, 50)
        self.assertEqual(default.picture_tokens(50), 327)
        with patch.dict(os.environ, {'TENSORFOLD_GLM_MAX_IMAGES': '0',
                                    'TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS': '0'}):
            unlimited = glm.GlmVisionLimits.from_env()
        self.assertEqual(unlimited.max_images, 0)
        self.assertEqual(unlimited.picture_tokens(51), 2048)
        self.assertEqual(unlimited.picture_tokens(1000), 2048)

    def test_count_enforced_unless_disabled(self):
        message = [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': URL}} for _ in range(51)]}]
        with self.assertRaises(ImageInputError):
            images.split_images(message, limits=images.ImageLimits(max_images=50))
        _, sources = images.split_images(message, limits=images.ImageLimits(max_images=0))
        self.assertEqual(len(sources), 51)
        self.assertEqual(len(images.load_images(sources, limits=images.ImageLimits(max_images=0))), 51)
        with self.assertRaises(ImageInputError):
            images.load_images(sources, limits=images.ImageLimits(max_images=50))

    def test_zero_not_permitted_for_resource_budgets(self):
        images.ImageLimits(max_images=0, max_redirects=0)
        for key in ('max_total_pixels', 'max_total_encoded_bytes', 'total_timeout_seconds'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                images.ImageLimits(**{key: 0})

    def test_uncapped_retains_default_glm_resource_budgets(self):
        old = frontend(glm.GlmVisionLimits()).image_limits
        new = frontend(glm.GlmVisionLimits(max_images=0)).image_limits
        self.assertEqual(new, replace(old, max_images=0))

    def test_uncapped_still_enforces_aggregate_bytes(self):
        limits = images.ImageLimits(max_images=0, max_total_encoded_bytes=len(DATA))
        with self.assertRaises(ImageInputError):
            images.load_images([SOURCE, SOURCE], limits=limits)

    def test_cached_images_still_count_toward_pixels(self):
        obj = frontend(glm.GlmVisionLimits(max_images=0, request_image_tokens=0))
        pixels = obj.image_limits.max_total_pixels // 2 + 1
        key = (hashlib.sha256(DATA).digest(), 'auto', 2048, 16)
        cached = SimpleNamespace(pixels=pixels)
        obj._pictures[key] = cached
        with patch.object(glm, 'PICTURE_CACHE', True):
            self.assertEqual(obj.load_images([SOURCE]), [cached])
            with self.assertRaisesRegex(ImageInputError, 'total pixel limit'):
                obj.load_images([SOURCE, SOURCE])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *rest])
