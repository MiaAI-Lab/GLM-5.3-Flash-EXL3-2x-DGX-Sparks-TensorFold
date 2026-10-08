"""0107: the prompt kernels' choices that set a prompt's bits (``forward.prompt_arith_code``) are in the ranks' startup
comparison and in the prompt cache's identity. On CPU, no model:

- the code changes with every choice that changes Flash's prompt bits (TF_GLM_PROMPT_PARTIALS, TF_GLM_MSA,
  TF_GLM_EXL3_PROMPT, TF_GLM_SPARSE_ONEPASS, TF_GLM_LATENT_MMA, TF_GLM_HC_MMA, TF_GLM_KDA_CHUNKED), one int each, and
  not with the ones that leave every bit as it was (TF_GLM_EXL3_PASS / _ORDER / _ROWROT, the
  split's exchange, overlap, own-row front and micro-batches);
- the startup comparison (``engine.odd_ranks`` on the rows the ranks gather) rejects ranks started with different
  settings, for each choice, and passes equal ones; the engine's row ends in the code and its error names them;
- the cache identity: the spill tier's folder (``SpillStore.compat``) is keyed on the engine's agreed settings, which
  hold the code, so fp32 and bf16 partials (or any other choice) never share a folder: a kept state is never resumed
  under other arithmetic than made it."""

import inspect

import pytest

from tensorfold.families.glm5_next.cuda import engine, forward, glue, kda, latent, msa
from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings

SPLIT = {"TF_GLM_HC_SPLIT": "1"}


def code(split_env=None):
    """prompt_arith_code with the split settings from ``split_env`` (the module switches as they are)."""
    return forward.prompt_arith_code(SplitSettings.from_env({**SPLIT, **(split_env or {})}))


@pytest.fixture
def switch(monkeypatch):
    """Set one of the choices: module constants, or TF_GLM_EXL3_PROMPT's environment variable."""
    def put(name, on):
        if name == "partials":
            return {"TF_GLM_PROMPT_PARTIALS": "bf16" if on else "fp32"}
        if name == "exl3":
            monkeypatch.setenv("TF_GLM_EXL3_PROMPT", "1" if on else "0")
        else:
            module, attr = {"msa": (msa, "ENABLED"), "onepass": (latent, "SPARSE_ONEPASS"),
                            "latent_mma": (latent, "PROMPT_MMA"), "hc_mma": (glue, "HC_MMA"),
                            "kda": (kda, "CHUNKED")}[name]
            monkeypatch.setattr(module, attr, on)
        return {}
    return put


CHOICES = ["partials", "msa", "exl3", "onepass", "latent_mma", "hc_mma", "kda"]


def test_code_is_ints_one_a_choice():
    got = code()
    assert all(isinstance(v, int) for v in got) and len(got) == len(CHOICES)
    assert forward.prompt_arith_code() == forward.prompt_arith_code(SplitSettings.from_env(SPLIT))   # fp32 by default
    assert code({"TF_GLM_PROMPT_PARTIALS": "bf16"}) != got


@pytest.mark.parametrize("name", CHOICES)
def test_code_changes_with_each_choice(switch, name):
    off, on = {}, {}
    off.update(switch(name, False))
    a = code(off)
    on.update(switch(name, True))
    b = code(on)
    assert a != b and len(a) == len(b)
    # only that choice's int moved
    assert sum(x != y for x, y in zip(a, b)) == 1


def test_code_ignores_choices_that_keep_every_bit(monkeypatch):
    base = code()
    for var, val in (("TF_GLM_EXL3_PASS", "128"), ("TF_GLM_EXL3_ORDER", "0"), ("TF_GLM_EXL3_ROWROT", "0")):
        monkeypatch.setenv(var, val)                                               # the same bits (their docstrings)
    assert code() == base
    for env in ({"TF_GLM_HC_EXCHANGE": "gather"}, {"TF_GLM_PREFILL_OVERLAP": "1"}, {"TF_GLM_PROMPT_OWN_FRONT": "1"},
                {"TF_GLM_PROMPT_MICROBATCH": "1"}, {"TF_GLM_HC_SPLIT_MIN_ROWS": "512"}):
        assert code(env) == base, env


@pytest.mark.parametrize("name", CHOICES)
def test_startup_comparison_rejects_mismatched_ranks(switch, name):
    def row(on):
        env = switch(name, on)
        # a rank's gathered row: whatever precedes, the code, then the spare memory the comparison leaves out
        return [1, 2, 3] + code(env) + [512]

    assert engine.odd_ranks([row(False), row(False), row(False)]) == []
    assert engine.odd_ranks([row(False), row(True)]) == [1]
    assert engine.odd_ranks([row(True), row(False), row(True), row(False)]) == [1, 3]
    # the spare memory alone never refuses a start
    assert engine.odd_ranks([row(False), row(False)[:-1] + [64]]) == []


def test_engine_row_ends_in_the_code_and_names_it():
    src = inspect.getsource(engine.GlmEngine.__init__)
    assert "+ prompt_arith_code(self.split)" in src and "odd = odd_ranks(rows)" in src
    for var in ("TF_GLM_PROMPT_PARTIALS", "TF_GLM_MSA", "TF_GLM_EXL3_PROMPT", "TF_GLM_SPARSE_ONEPASS",
                "TF_GLM_LATENT_MMA", "TF_GLM_HC_MMA", "TF_GLM_KDA_CHUNKED"):
        assert var in src, var
    # what the spill tier's folders are keyed on (everything but the window's slots) holds the code at its end
    assert "self.agreed = mine[:1] + mine[2:]" in src


def test_cache_identity_is_keyed_on_the_agreed_settings():
    """Both the single-stream engine's spill store and the multi-stream pool's signature hold the agreed settings."""
    assert "repr(self.agreed)" in inspect.getsource(engine.GlmEngine._spill_store)
    from tensorfold.families.glm5_next.cuda import multi

    assert "repr(self.g.agreed)" in inspect.getsource(multi)


@pytest.mark.parametrize("name", CHOICES)
def test_cache_identity_differs(tmp_path, switch, name):
    from tensorfold.cuda.spill import SpillConfig, SpillStore

    def folder(on):
        env = switch(name, on)
        mine = [1, 2048, 3, 4] + code(env)                  # the engine's row: [:1] and [2:] are the agreed settings
        agreed = mine[:1] + mine[2:]
        sig = "|".join(["single", "build", "", "", "tp2", "[]", repr(agreed)])
        store = SpillStore(SpillConfig(str(tmp_path), 1.0, min_free_gib=0.0), rank=0, world=1, device="cpu",
                           signature=sig, weights="w", model_id="m", quiet=True)
        return store.compat

    off, on = folder(False), folder(True)
    assert off != on, f"{name}: a state kept under one arithmetic would resume under the other"
    assert folder(False) == off and folder(True) == on
