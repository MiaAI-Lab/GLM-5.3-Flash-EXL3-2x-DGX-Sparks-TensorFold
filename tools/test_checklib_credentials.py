"""tools/checklib.py reads the server's API key only from API_KEY, TENSORFOLD_API_KEY (the variable this recipe passes
to the server) or an explicit --api-key-file, and never writes it.

The DeepSeek recipe's rule (maintainer review of MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#30): no default path
(no home folder, no API_KEY_FILE variable), nothing but the Authorization header carries the key, receipts and printed
lines are scrubbed. Run: python3 -I tools/test_checklib_credentials.py (also part of tools/selftest.py).
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checklib as C  # noqa: E402

KEY = "sk-tf-test-0123456789abcdef"


class CredentialTest(unittest.TestCase):
    def setUp(self):
        C._api_key_file = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, C, "_api_key_file", None)

    def keyfile(self, text):
        p = Path(self.tmp.name) / "api-key"
        p.write_text(text)
        return str(p)

    def parse(self, *argv):
        return C.use_api_key_args(C.add_api_key_arg(argparse.ArgumentParser()).parse_args(list(argv)))

    def test_no_default_path_and_no_api_key_file_variable(self):
        home = Path(self.tmp.name) / "home"
        state = home / ".local/state/glm53-tensorfold"
        state.mkdir(parents=True)
        (state / "api-key").write_text(KEY + "\n")
        env = {"HOME": str(home), "API_KEY_FILE": str(state / "api-key")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.parse()
            self.assertEqual(C.api_key(), "")
            self.assertNotIn("Authorization", C.headers())

    def test_api_key_variable(self):
        with mock.patch.dict(os.environ, {"API_KEY": " " + KEY + " "}, clear=True):
            self.parse()
            self.assertEqual(C.api_key(), KEY)
            self.assertEqual(C.headers()["Authorization"], "Bearer " + KEY)
        for switch in ("0", "1", ""):   # start.sh's API_KEY switch values are not keys
            with mock.patch.dict(os.environ, {"API_KEY": switch}, clear=True):
                self.assertEqual(C.api_key(), "")

    def test_server_variable(self):
        with mock.patch.dict(os.environ, {"TENSORFOLD_API_KEY": " , " + KEY + ",sk-second"}, clear=True):
            self.parse()
            self.assertEqual(C.api_key(), KEY)       # the first of the server's comma-separated keys
        with mock.patch.dict(os.environ, {"TENSORFOLD_API_KEY": KEY, "API_KEY": "sk-env"}, clear=True):
            self.assertEqual(C.api_key(), "sk-env")   # API_KEY wins
        with mock.patch.dict(os.environ, {"TENSORFOLD_API_KEY": KEY, "API_KEY": "1"}, clear=True):
            self.assertEqual(C.api_key(), KEY)        # a switch value of API_KEY is not a key
        with mock.patch.dict(os.environ, {"TENSORFOLD_API_KEY": KEY}, clear=True):
            self.parse("--api-key-file", self.keyfile("sk-file"))
            self.assertEqual(C.api_key(), KEY)        # the server's variable before the file

    def test_explicit_file(self):
        f = self.keyfile("# a comment\n\n" + KEY + "\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.parse("--api-key-file", f)
            self.assertEqual(C.api_key(), KEY)
        f = self.keyfile("Authorization: Bearer-less " + KEY + "\n")   # a header line: the value after the colon
        with mock.patch.dict(os.environ, {}, clear=True):
            self.parse("--api-key-file", f)
            self.assertEqual(C.api_key(), "Bearer-less " + KEY)
        with mock.patch.dict(os.environ, {"API_KEY": "sk-env"}, clear=True):   # the variable wins
            self.parse("--api-key-file", self.keyfile(KEY))
            self.assertEqual(C.api_key(), "sk-env")

    def test_unreadable_explicit_file_is_an_error_without_the_path_contents(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.parse("--api-key-file", str(Path(self.tmp.name) / "missing"))
            with self.assertRaises(SystemExit) as e:
                C.api_key()
            self.assertIn("cannot read it", str(e.exception))

    def test_receipts_and_lines_never_carry_the_key(self):
        out = Path(self.tmp.name) / "receipt.json"
        with mock.patch.dict(os.environ, {"API_KEY": KEY}, clear=True):
            r = C.Receipt("t", {"note": "url with " + KEY})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r.add("c1", "FAIL", "server said: bad key " + KEY, body={"echo": [KEY, 3]})
                r.finish(str(out))
        self.assertNotIn(KEY, buf.getvalue())
        text = out.read_text()
        self.assertNotIn(KEY, text)
        doc = json.loads(text)
        self.assertEqual(doc["checks"][0]["body"]["echo"], ["<api-key>", 3])

    def test_every_checklib_tool_takes_the_flag(self):
        tools = Path(__file__).resolve().parent
        for name in ("spill_check.py", "resume_bench.py", "prefix_fanout.py", "exact.py"):
            with self.subTest(tool=name):
                src = (tools / name).read_text()
                self.assertIn("C.use_api_key_args(C.add_api_key_arg(ap).parse_args())", src)
                self.assertNotIn("API_KEY_FILE", src)
        self.assertNotIn("expanduser", (tools / "checklib.py").read_text())


if __name__ == "__main__":
    unittest.main()
