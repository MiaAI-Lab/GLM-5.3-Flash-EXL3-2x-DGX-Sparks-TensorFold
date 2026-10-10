"""The opt-in preset changes only four existing settings and preserves overrides."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


PRESET = dict(TF_GLM_EXL3_DEC_ORDER="2", TF_GLM_MULTI_SAMPLER="packed",
              TF_GLM_MULTI_DEPTH="joint", TF_GLM_MULTI_ASYNC="1")


def config(tmp_path, **settings):
    source = Path(__file__).resolve().parents[2] / "scripts/config.sh"
    if not source.exists():
        source = Path("/recipe/scripts/config.sh")
    folder = tmp_path / "scripts"
    folder.mkdir(exist_ok=True)
    shutil.copy2(source, folder / "config.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TP", "TENSORFOLD_"))}
    env.update(settings)
    result = subprocess.run(["bash", "-c", 'source "$1" && env -0', "_", str(folder / "config.sh")],
                            env=env, text=True, capture_output=True)
    return result, dict(row.split("=", 1) for row in result.stdout.split("\0") if "=" in row)


@pytest.mark.parametrize("tp", ["2", "3", "4"])
@pytest.mark.parametrize("flag", [None, "0"])
def test_default_is_unchanged(tmp_path, tp, flag):
    settings = {} if flag is None else dict(TP4_DECODE_TUNING=flag)
    result, env = config(tmp_path, TP=tp, **settings)
    assert result.returncode == 0, result.stderr
    assert env["TF_GLM_EXL3_DEC_ORDER"] == "0"
    assert all(k not in env for k in PRESET if k != "TF_GLM_EXL3_DEC_ORDER")
    assert env["TF_GLM_SIDE"] == ("1" if tp == "4" else "0")


def test_only_four_settings_change(tmp_path):
    result, before = config(tmp_path, TP="4")
    assert result.returncode == 0, result.stderr
    result, after = config(tmp_path, TP="4", TP4_DECODE_TUNING="1")
    assert result.returncode == 0, result.stderr
    changed = {k: after.get(k) for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == dict(PRESET, TP4_DECODE_TUNING="1")
    assert after["TF_GLM_SIDE"] == "1"
    assert all(k not in after for k in ("TF_GLM_EXL3_DEC_PRMT", "TF_GLM_QMM_CLUSTERS", "TF_GLM_SEG_CHUNKS_CUDA"))


@pytest.mark.parametrize("key,value", [("TF_GLM_EXL3_DEC_ORDER", "0"), ("TF_GLM_MULTI_SAMPLER", "streams"),
                                      ("TF_GLM_MULTI_DEPTH", "policy"), ("TF_GLM_MULTI_ASYNC", "0")])
def test_individual_override_wins(tmp_path, key, value):
    result, env = config(tmp_path, TP="4", TP4_DECODE_TUNING="1", **{key: value})
    assert result.returncode == 0, result.stderr
    assert {k: env[k] for k in PRESET} == dict(PRESET, **{key: value})


@pytest.mark.parametrize("settings", [{"TP": "2", "TP4_DECODE_TUNING": "1"},
                                     {"TP": "3", "TP4_DECODE_TUNING": "1"},
                                     {"TP": "4", "TP4_DECODE_TUNING": "bad"}])
def test_invalid_preset(tmp_path, settings):
    result, _ = config(tmp_path, **settings)
    assert result.returncode != 0 and "TP4_DECODE_TUNING" in result.stderr
