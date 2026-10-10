"""Opt-in TP4 preset, explicit overrides, and safe CUDA-attention fallback."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


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
def test_default_is_unchanged(tmp_path, tp):
    result, env = config(tmp_path, TP=tp)
    assert result.returncode == 0, result.stderr
    assert "TF_GLM_EXL3_DEC_PRMT" not in env
    assert "TF_GLM_QMM_CLUSTERS" not in env
    assert "TF_GLM_SEG_CHUNKS_CUDA" not in env
    assert "TF_GLM_MULTI_ASYNC" not in env
    assert env["TF_GLM_SIDE"] == ("1" if tp == "4" else "0")


def test_preset_and_overrides(tmp_path):
    result, env = config(tmp_path, TP="4", TP4_DECODE_TUNING="1")
    assert result.returncode == 0, result.stderr
    assert env["TF_GLM_EXL3_DEC_PRMT"] == env["TF_GLM_SEG_CHUNKS_CUDA"] == "1"
    assert env["TF_GLM_MULTI_ASYNC"] == "1" and env["TF_GLM_MULTI_SAMPLER"] == "packed"
    assert env["TF_GLM_MULTI_DEPTH"] == "joint" and env["TF_GLM_QMM_CLUSTERS"] == "0"
    assert env["TF_GLM_SIDE"] == "0" and env["TF_GLM_EXL3_DEC_ORDER"] == "2"
    result, env = config(tmp_path, TP="4", TP4_DECODE_TUNING="1", TF_GLM_SIDE="1",
                         TF_GLM_EXL3_DEC_PRMT="0", TF_GLM_QMM_CLUSTERS="1", TF_GLM_MULTI_ASYNC="0")
    assert result.returncode == 0 and env["TF_GLM_SIDE"] == "1"
    assert env["TF_GLM_EXL3_DEC_PRMT"] == env["TF_GLM_MULTI_ASYNC"] == "0"
    assert env["TF_GLM_QMM_CLUSTERS"] == "1"


@pytest.mark.parametrize("settings", [{"TP": "2", "TP4_DECODE_TUNING": "1"},
                                     {"TP": "4", "TP4_DECODE_TUNING": "bad"}])
def test_invalid_preset(tmp_path, settings):
    result, _ = config(tmp_path, **settings)
    assert result.returncode != 0 and "TP4_DECODE_TUNING" in result.stderr


def test_attention_fails_closed(monkeypatch):
    from tensorfold.families.glm5_next.cuda import latent
    monkeypatch.setattr(latent, "_SEG_CHUNKS_EXT", None)
    monkeypatch.setattr(latent, "_SEG_CHUNKS_FAILED", None)
    monkeypatch.delenv("TF_GLM_SEG_CHUNKS_CUDA", raising=False)
    assert not latent.seg_chunks_cuda()
    monkeypatch.setenv("TF_GLM_SEG_CHUNKS_CUDA", "1")
    assert latent.seg_chunks_cuda()
    monkeypatch.setattr(latent.triton, "__version__", "unsupported")
    assert latent._seg_chunks_ext() is None
    assert "requires Triton" in latent._SEG_CHUNKS_FAILED
    assert not latent.seg_chunks_cuda()


def test_prmt_build_identity(monkeypatch):
    from tensorfold.cuda import build
    from tensorfold.families.glm5_next.cuda import exl3_mm
    calls = []
    monkeypatch.setattr(build, "load", lambda **kw: calls.append(kw) or object())
    try:
        for flag in ("0", "1"):
            exl3_mm._ext.cache_clear()
            monkeypatch.setenv("TF_GLM_EXL3_DEC_PRMT", flag)
            exl3_mm._ext()
            assert f"-DTF_GLM_DECODE_PRMT={flag}" in calls[-1]["extra_cuda_cflags"]
        assert calls[0]["name"] != calls[1]["name"]
        exl3_mm._ext.cache_clear()
        monkeypatch.setenv("TF_GLM_EXL3_DEC_PRMT", "invalid")
        with pytest.raises(ValueError):
            exl3_mm._ext()
    finally:
        exl3_mm._ext.cache_clear()
