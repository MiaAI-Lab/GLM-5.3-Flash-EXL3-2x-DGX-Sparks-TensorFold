"""TP4 defaults select four existing settings and preserve configuration precedence."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tensorfold.families.glm5_next.cuda.multi_tune import decode_defaults, MultiSettings


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
    exported = dict(row.split("=", 1) for row in result.stdout.split("\0") if "=" in row)
    decode_defaults(int(exported.get("TP", "2")), exported)
    return result, exported


@pytest.mark.parametrize("tp", [None, "2", "3"])
def test_other_tp_defaults_unchanged(tmp_path, tp):
    result, env = config(tmp_path, **({} if tp is None else dict(TP=tp)))
    assert result.returncode == 0, result.stderr
    assert env.get("TF_GLM_EXL3_DEC_ORDER", "0") == "0"
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
    result, env = config(tmp_path, TP="4", **{source: f"{key}={value}\n"})
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


@pytest.mark.parametrize("tp", ["2", "3"])
def test_other_tp_empty_order_keeps_recipe_fallback(tmp_path, tp):
    result, env = config(tmp_path, TP=tp, TF_GLM_EXL3_DEC_ORDER="")
    assert result.returncode == 0, result.stderr
    assert env["TF_GLM_EXL3_DEC_ORDER"] == "0"


@pytest.mark.parametrize("world", [2, 3, 4])
def test_engine_defaults_are_bounded_and_idempotent(world):
    env = {"unrelated": "keep", "TF_GLM_SIDE": "1", "TF_GLM_MULTI_PROFILE": "0"}
    before = env.copy()
    decode_defaults(world, env)
    assert env == (dict(before, **PRESET) if world == 4 else before)
    decode_defaults(world, env)
    assert env == (dict(before, **PRESET) if world == 4 else before)


@pytest.mark.parametrize("key", list(PRESET))
def test_empty_value_uses_tp4_default(key):
    env = {key: ""}
    decode_defaults(4, env)
    assert env == PRESET


@pytest.mark.parametrize("key", list(PRESET))
def test_invalid_override_is_not_silently_replaced(monkeypatch, key):
    from tensorfold.families.glm5_next.cuda.exl3_mm import dec_order

    env = {key: "invalid"}
    decode_defaults(4, env)
    assert env[key] == "invalid"
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(ValueError, match=key):
        if key == "TF_GLM_EXL3_DEC_ORDER":
            dec_order()
        else:
            MultiSettings.from_env()


@pytest.mark.parametrize("world,rank,comm_world", [(2, 0, None), (3, 0, None),
                         (4, 0, None), (4, 1, None), (4, 2, None), (4, 3, None), (2, 0, 4)])
def test_constructor_defaults_before_cuda_and_settings_read(monkeypatch, world, rank, comm_world):
    """Run the real constructor up to the first CUDA call, without a GPU or model."""
    from types import SimpleNamespace
    import torch
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.exl3_mm import dec_order

    env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TENSORFOLD_"))}
    monkeypatch.setattr(os, "environ", env)

    class BeforeCUDA(Exception):
        pass

    def stop_before_cuda(device):
        raise BeforeCUDA

    monkeypatch.setattr(torch.cuda, "set_device", stop_before_cuda)
    comm = None if comm_world is None else SimpleNamespace(world=comm_world)
    with pytest.raises(BeforeCUDA):
        GlmEngine(Path("/no-model-needed"), rank=rank, world=world, master="unused", port=1,
                  serial_only=True, comm=comm)
    tp4 = (comm_world if comm_world is not None else world) == 4
    assert {k: env[k] for k in PRESET if k in env} == (PRESET if tp4 else {})
    expected = MultiSettings.from_env(PRESET if tp4 else {})
    assert MultiSettings.from_env().code() == expected.code()
    assert dec_order() == (2 if tp4 else 0)
