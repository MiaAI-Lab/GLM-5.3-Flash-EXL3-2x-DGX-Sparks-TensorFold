"""Offline tests. Fixtures are never reported as inference results."""
import importlib.util
import io
import contextlib
from pathlib import Path
import unittest
from unittest.mock import patch

p=Path(__file__).resolve().parents[1]/'tools/hermes_benchmark.py'
s=importlib.util.spec_from_file_location('benchmark',p)
assert s is not None and s.loader is not None
b=importlib.util.module_from_spec(s)
s.loader.exec_module(b)

class HarnessTests(unittest.TestCase):
    def test_import_does_not_run_inference(self):
        self.assertIsNone(b.MODEL)
        self.assertEqual(b.BASE,'http://127.0.0.1:8888/v1')

    def test_unrecognized_model_rejected_before_network(self):
        with patch.object(b.opener,'open') as request:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    b.main(['--model','wrong-model'])
            request.assert_not_called()

    def test_exact_matched_prompt_set(self):
        self.assertEqual(len(b.jobs),8)
        self.assertEqual(len(set(n for n,_,_ in b.jobs)),8)
        self.assertEqual(b.jobs[0][1],b.jobs[3][1])
        self.assertEqual(b.jobs[1][1],b.jobs[2][1])
        self.assertTrue(b.jobs[5][2])

    def test_wrong_response_model_is_rejected(self):
        rows=[b'data: {"model":"wrong","choices":[{"delta":{"content":"fixture"}}]}\n', b'data: [DONE]\n']
        with patch.object(b,'jobs',[('fixture','fixture',False)]), patch.object(b,'get_health',return_value={'backend':'tensorfold','busy':False}), patch.object(b,'memory',return_value=99), patch.object(b.opener,'open') as request:
            request.return_value.__enter__.return_value=iter(rows)
            with self.assertRaises(AssertionError):
                b.main(['--model','GLM-5.3-Flash-EXL3'])
        setattr(b,'MODEL',None)

if __name__=='__main__': unittest.main()
