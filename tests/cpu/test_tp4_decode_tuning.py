"""TP4 defaults select four existing settings and preserve configuration precedence."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


PRESET = dict(TF_GLM_EXL3_DEC_ORDER="2", TF_GLM_MULTI_SAMPLER="packed",
              TF_GLM_MULTI_DEPTH="joint", TF_GLM_MULTI_ASYNC="1")


def config(tmp_path, local=None, dotenv=None, **settings):
    source = Path(__file__).resolve().parents[2] / "scripts/config.sh"
    if not source.exists():
        source = Path("/recipe/scripts/config.sh")
    folder = tmp_path / "scripts"
    folder.mkdir(exist_ok=True)
    shutil.copy2(source, folder / "config.sh")
    if local is not None:
        (folder / "local.sh").write_text(local)
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TP", "TENSORFOLD_"))}
    env.update(settings)
    result = subprocess.run(["bash", "-c", 'source "$1" && env -0', "_", str(folder / "config.sh")],
                            env=env, text=True, capture_output=True)
    return result, dict(row.split("=", 1) for row in result.stdout.split("\0") if "=" in row)


@pytest.mark.parametrize("tp", [None, "2", "3"])
def test_other_tp_defaults_unchanged(tmp_path, tp):
    result, env = config(tmp_path, **({} if tp is None else dict(TP=tp)))
    assert result.returncode == 0, result.stderr
    assert env["TF_GLM_EXL3_DEC_ORDER"] == "0"
    assert all(k not in env for k in PRESET if k != "TF_GLM_EXL3_DEC_ORDER")
    assert env["TF_GLM_SIDE"] == "0"


def test_only_four_settings_change(tmp_path):
    previous = dict(TF_GLM_EXL3_DEC_ORDER="0", TF_GLM_MULTI_SAMPLER="streams",
                    TF_GLM_MULTI_DEPTH="policy", TF_GLM_MULTI_ASYNC="0")
    result, before = config(tmp_path, TP="4", **previous)
    assert result.returncode == 0, result.stderr
    result, after = config(tmp_path, TP="4")
    assert result.returncode == 0, result.stderr
    changed = {k: after.get(k) for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == PRESET
    assert after["TF_GLM_SIDE"] == "1"
    assert all(k not in after for k in ("TF_GLM_EXL3_DEC_PRMT", "TF_GLM_QMM_CLUSTERS", "TF_GLM_SEG_CHUNKS_CUDA"))


@pytest.mark.parametrize("key,value", [("TF_GLM_EXL3_DEC_ORDER", "0"), ("TF_GLM_MULTI_SAMPLER", "streams"),
                                      ("TF_GLM_MULTI_DEPTH", "policy"), ("TF_GLM_MULTI_ASYNC", "0")])
def test_individual_override_wins(tmp_path, key, value):
    result, env = config(tmp_path, TP="4", **{key: value})
    assert result.returncode == 0, result.stderr
    assert {k: env[k] for k in PRESET} == dict(PRESET, **{key: value})


@pytest.mark.parametrize("key,value", [("TF_GLM_EXL3_DEC_ORDER", "0"), ("TF_GLM_MULTI_SAMPLER", "streams"),
                                      ("TF_GLM_MULTI_DEPTH", "policy"), ("TF_GLM_MULTI_ASYNC", "0")])
@pytest.mark.parametrize("source", ["local", "dotenv"])
def test_file_override_wins(tmp_path, key, value, source):
    result, env = config(tmp_path, TP="4", **{source: f"export {key}={value}\n"})
    assert result.returncode == 0, result.stderr
    assert {k: env[k] for k in PRESET} == dict(PRESET, **{key: value})


@pytest.mark.parametrize("environment_wins", [False, True])
def test_configuration_precedence(tmp_path, environment_wins):
    settings = dict(TF_GLM_EXL3_DEC_ORDER="0") if environment_wins else {}
    result, env = config(tmp_path, TP="4", local="export TF_GLM_EXL3_DEC_ORDER=1\n",
                         dotenv="TF_GLM_EXL3_DEC_ORDER=2\n", **settings)
    assert result.returncode == 0, result.stderr
    assert env["TF_GLM_EXL3_DEC_ORDER"] == ("0" if environment_wins else "1")


@pytest.mark.parametrize("tp", ["2", "3"])
def test_other_tp_explicit_settings_preserved(tmp_path, tp):
    result, env = config(tmp_path, TP=tp, **PRESET)
    assert result.returncode == 0, result.stderr
    assert {k: env[k] for k in PRESET} == PRESET
    assert env["TF_GLM_SIDE"] == "0"
