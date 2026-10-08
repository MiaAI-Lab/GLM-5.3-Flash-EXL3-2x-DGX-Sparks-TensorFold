"""0100: the RoCE proxy's queue pairs wait for an ACK as NCCL does: timeout 20 unless TF_ROCE_IB_TIMEOUT (1-31) says
otherwise. roce_proxy.c is compiled here with a small driver that calls its timeout rule (no RDMA device needed)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DRIVER = r'''
#include "roce_proxy.c"
int main(void) { printf("%d\n", (int)roce_ib_timeout()); return 0; }
'''


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    import tensorfold.cuda as pkg

    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    src = Path(pkg.__file__).parent / "roce_proxy.c"
    d = tmp_path_factory.mktemp("roce")
    (d / "driver.c").write_text(DRIVER)
    out = d / "driver"
    r = subprocess.run(["gcc", "-O1", "-I", str(src.parent), str(d / "driver.c"), "-o", str(out), "-libverbs",
                        "-lpthread"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"cannot build against libibverbs here: {r.stderr[-300:]}")
    return out


def timeout(driver, value=None) -> int:
    env = {k: v for k, v in os.environ.items() if k != "TF_ROCE_IB_TIMEOUT"}
    if value is not None:
        env["TF_ROCE_IB_TIMEOUT"] = value
    return int(subprocess.run([str(driver)], env=env, capture_output=True, text=True, check=True).stdout)


def test_default_is_nccls(driver):
    assert timeout(driver) == 20
    assert timeout(driver, "") == 20


@pytest.mark.parametrize("value,want", [("14", 14), ("1", 1), ("31", 31), ("0", 20), ("32", 20), ("-3", 20),
                                        ("x", 20)])
def test_override(driver, value, want):
    assert timeout(driver, value) == want
