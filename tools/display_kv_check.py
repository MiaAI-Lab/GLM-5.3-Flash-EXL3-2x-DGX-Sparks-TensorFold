#!/usr/bin/env python3
"""Checks for patches 0072 (DISPLAY_KV_MIB) and 0087 (DISPLAY_KV_BACKEND=dispram), run inside the image that
scripts/prepare.sh built:

    docker run --rm --gpus all --entrypoint python -v "$PWD/tools/display_kv_check.py:/c.py" tensorfold-glm53:v0.6.0 \
      /c.py [--gpu]

Without --gpu: the setting, the refusal while a display is connected, how many latent planes the span takes, the span's mapping and unwinding (a fake for every
DRM and CUDA driver call), the planes carved from a span, the pool's row copies of spanned planes against stock
ones, and 0087's backend switch and refusals (a fake dispram client). With --gpu, on a Spark whose display reservation
is free (the server stopped, or started with DISPLAY_KV_MIB=0): the real span, written and read back by the GPU across
both halves, and a pool copy across the boundary. On kindling spark-os, run the GPU half through dispramd: add
-e TF_GLM_DISPLAY_KV_BACKEND=dispram and start.sh's mounts (-v /run/dispram:/run/dispram -v
/opt/kindling/dispram/python:/opt/dispram:ro -e PYTHONPATH=/opt/dispram). Exit code 1 when a check fails.
"""
import os
import sys

import torch

from tensorfold.families.glm5_next.cuda import display_kv as dk
from tensorfold.families.glm5_next.cuda import kv8

MIB = 2 ** 20
failures = []
# the backend the --gpu half maps the real span through (patch 0087); the fake driver checks below use drm
GPU_BACKEND = os.environ.pop(dk.BACKEND_ENV, None)


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(call, *args, match=""):
    try:
        call(*args)
    except (ValueError, RuntimeError) as exc:
        return match in str(exc)
    return False

# patch 0087: with DISPLAY_KV_BACKEND unset the span is the DRM path's, which every fake driver check below drives
check("the backend is drm unless DISPLAY_KV_BACKEND says otherwise", dk.backend() == "drm")


# -- the setting --------------------------------------------------------------------------------------------------------
def setting(value):
    if value is None:
        os.environ.pop(dk.ENV, None)
    else:
        os.environ[dk.ENV] = value
    return dk.mib()


check("unset is off", setting(None) == 0)
check("0 is off", setting("0") == 0)
check("1792 MiB", setting("1792") == 1792)
check("credit in bytes", dk.credit() == 1792 * MIB)
check("2032 is the largest", setting("2032") == 2032)
for bad in ("2048", "17", "-16", "abc"):
    check(f"{bad} refused", raises(setting, bad, match=dk.ENV))
setting(None)
check("no credit when off", dk.credit() == 0)

# -- how many latent planes the span takes ------------------------------------------------------------------------------
check("two planes cover 1792 MiB of 1 GB planes", dk.span_planes(17, 10 ** 9, 1792 * MIB) == 2)
check("one plane when it is larger", dk.span_planes(17, 2 * 10 ** 9, 1792 * MIB) == 1)
check("exact fit", dk.span_planes(4, 1792 * MIB // 2, 1792 * MIB) == 2)
check("too few planes refused", raises(dk.span_planes, 1, 10 ** 9, 1792 * MIB, match="latent planes"))


# -- mapping the span: every driver call through the seam, unwound on failure ---------------------------------------
class FakeSystem:
    fail = None             # (call, nth): the nth call of that name raises
    dumb_bytes = None       # None: what was asked
    delta = None            # None: the right one
    last = None

    def __init__(self):
        self.calls = []
        FakeSystem.last = self

    def _do(self, name, *args):
        self.calls.append((name, *args))
        if FakeSystem.fail and FakeSystem.fail == (name, sum(c[0] == name for c in self.calls)):
            raise OSError(f"{name} failed on purpose")

    def open_card(self):
        self._do("open_card")
        return 7

    def create_dumb(self, fd, size):
        self._do("create_dumb", size)
        return 3, size if FakeSystem.dumb_bytes is None else FakeSystem.dumb_bytes

    def map_offset(self, fd, handle):
        self._do("map_offset")
        return 1 << 40

    def reserve(self, size):
        self._do("reserve", size)
        return 1 << 44

    def map_ordinary(self, address, size):
        self._do("map_ordinary", address, size)

    def map_display(self, address, size, fd, offset):
        self._do("map_display", address, size)

    def register(self, address, size, *, io_memory):
        self._do("register", address, size, io_memory)

    def device_pointer(self, address):
        self._do("device_pointer", address)
        if FakeSystem.delta is not None and address != 1 << 44:
            return (1 << 44) + FakeSystem.delta
        return address

    def unregister(self, address):
        self._do("unregister", address)

    def unmap(self, address, size):
        self._do("unmap", address, size)

    def close_gem(self, fd, handle):
        self._do("close_gem")

    def close_fd(self, fd):
        self._do("close_fd")


def names(system):
    return [c[0] for c in system.calls]


def fake_sysfs(states):
    import tempfile

    root = tempfile.mkdtemp()
    for name, state in states.items():
        os.makedirs(f"{root}/{name}")
        with open(f"{root}/{name}/status", "w") as f:
            f.write(state + "\n")
    return root


# -- a connected display keeps its reservation ------------------------------------------------------------------------
check("headless: no output of card0 connected",
      dk.connected_outputs(fake_sysfs({"card0-HDMI-A-1": "disconnected", "card0-Unknown-1": "disconnected",
                                       "card1-DP-1": "connected"}), "card0") == [])
check("a connected output is named",
      dk.connected_outputs(fake_sysfs({"card0-HDMI-A-1": "connected", "card0-DP-1": "disconnected"}), "card0")
      == ["HDMI-A-1"])
dk.SYSTEM = FakeSystem
D, O = 1792 * MIB, 256 * MIB
FakeSystem.last = None
dk.SYSFS = fake_sysfs({"card0-DP-2": "connected"})
check("a connected display refuses the span before any driver call",
      raises(dk.map_span, O, D, match="card0-DP-2") and FakeSystem.last is None)
dk.SYSFS = fake_sysfs({"card0-DP-2": "disconnected"})
span = dk.map_span(O, D)
check("span covers both halves", (span.pointer, span.size) == (1 << 44, O + D))
check("display half sits right above the ordinary half",
      ("map_display", (1 << 44) + O, D) in FakeSystem.last.calls)
check("both halves registered, the display half as I/O memory",
      ("register", 1 << 44, O, False) in FakeSystem.last.calls and
      ("register", (1 << 44) + O, D, True) in FakeSystem.last.calls)
span = dk.map_span(0, D)
check("no ordinary half when the planes are the display reservation exactly",
      "map_ordinary" not in names(FakeSystem.last) and span.size == D)

for step, nth in (("open_card", 1), ("create_dumb", 1), ("map_offset", 1), ("reserve", 1), ("map_ordinary", 1),
                  ("map_display", 1), ("register", 1), ("register", 2), ("device_pointer", 1), ("device_pointer", 2)):
    FakeSystem.fail = (step, nth)
    ok = raises(dk.map_span, O, D)
    s = FakeSystem.last
    succeeded = lambda name: name in names(s) and name != step                  # noqa: E731
    registered = [c[1] for c in s.calls if c[0] == "register"][:-1 if step == "register" else None]
    unwound = (all(("unregister", a) in s.calls for a in registered) and
               len(registered) == sum(c[0] == "unregister" for c in s.calls) and
               (("unmap", 1 << 44, O + D) in s.calls) == succeeded("reserve") and
               ("close_gem" in names(s)) == succeeded("create_dumb") and
               ("close_fd" in names(s)) == succeeded("open_card"))
    check(f"{step} #{nth} failing raises and unwinds", ok and unwound, str(s.calls))
FakeSystem.fail = None
FakeSystem.dumb_bytes = D - 4096
check("a short display reserve is refused", raises(dk.map_span, O, D, match="display reserve gave"))
FakeSystem.dumb_bytes = None
FakeSystem.delta = O + 4096
check("halves apart on the device are refused", raises(dk.map_span, O, D, match="apart"))
check("and unwound", ("unmap", 1 << 44, O + D) in FakeSystem.last.calls and "close_fd" in names(FakeSystem.last))
FakeSystem.delta = None

# -- latent planes carved from one span --------------------------------------------------------------------------------
dk.map_span = lambda ordinary, display: dk.Span(0, ordinary + display)        # noqa: E731
dk.as_tensor = lambda span, device: torch.full((span.size,), 7, dtype=torch.uint8)   # noqa: E731
rows, width = 1 << 20, 512
want = kv8.zeros(rows, width, "fp8", "meta")
plane_bytes = want.numel() * want.element_size()
display = 1792 * MIB
planes = dk.latent_planes(17, rows, width, "fp8", "cpu", display)
k = len(planes)
check("as many planes as the display reservation needs", k == dk.span_planes(17, plane_bytes, display), str(k))
check("each plane is the stock plane's shape and type",
      all(p.shape == want.shape and p.dtype == want.dtype for p in planes))
check("each plane zeroed", all(int(p.count_nonzero()) == 0 for p in planes))
base = planes[0].data_ptr()
check("planes back to back in one buffer",
      all(p.data_ptr() == base + i * plane_bytes for i, p in enumerate(planes)))
check("a second claim is refused", raises(dk.latent_planes, 17, rows, width, "fp8", "cpu", display, match="already"))
dk._OWNERS.clear()
wb = kv8.zeros(rows, width, "bf16", "meta")
planes = dk.latent_planes(17, rows, width, "bf16", "cpu", display)
check("bf16 planes too", all(p.shape == wb.shape and p.dtype == wb.dtype for p in planes))
dk._OWNERS.clear()

# -- copies between pool rows of spanned planes ---------------------------------------------------------------------
# CUDA refuses a memcpy whose range crosses from one registration into the other (the ordinary half into the display
# half), so a spanned plane's rows move by a kernel; every other plane copies as before. Same rows either way.
from tensorfold.families.glm5_next.cuda.pool import Arena, Plane                                       # noqa: E402

gen = torch.Generator().manual_seed(0)
for dtype, shape in ((torch.uint8, (64, 528)), (torch.bfloat16, (64, 512))):
    for src, dst, n in ((0, 32, 16), (4, 10, 30), (30, 2, 25), (0, 0, 8), (60, 1, 4)):
        base = torch.randint(0, 255, shape, generator=gen, dtype=torch.uint8) if dtype == torch.uint8 else \
            torch.randn(shape, generator=gen).to(dtype)
        pooled = torch.randint(0, 255, (64 // 4 + 2, 144), generator=gen, dtype=torch.uint8)
        stock = Arena(64, [Plane(base.clone()), Plane(pooled.clone(), 4, 2)])
        spanned = Arena(64, [Plane(base.clone(), spanned=True), Plane(pooled.clone(), 4, 2, spanned=True)])
        kernel = torch.bitwise_or
        calls = []
        torch.bitwise_or = lambda *a, **k: calls.append(1) or kernel(*a, **k)    # noqa: E731
        try:
            stock.copy(src, dst, n)
            stock_calls = len(calls)
            spanned.copy(src, dst, n)
        finally:
            torch.bitwise_or = kernel
        same = all(torch.equal(a.tensor, b.tensor) for a, b in zip(stock.planes, spanned.planes))
        by_kernel = stock_calls == 0 and (len(calls) > 0) == (n > 0 and src != dst)
        check(f"spanned rows copy as stock, by a kernel: {dtype} {src}->{dst} x{n}", same and by_kernel,
              f"stock kernel calls {stock_calls}, spanned {len(calls) - stock_calls}")

# -- patch 0087: the reservation from kindling's dispramd (DISPLAY_KV_BACKEND=dispram) --------------------------------
import importlib                                                                                         # noqa: E402

importlib.reload(dk)                        # the real map_span again (the sections above replaced it)


def backend(value):
    if value is None:
        os.environ.pop(dk.BACKEND_ENV, None)
    else:
        os.environ[dk.BACKEND_ENV] = value
    return dk.backend()


check("backend drm by default", backend(None) == "drm")
check("backend dispram, case and spaces as typed", backend("dispram") == "dispram" and backend(" DisPram ") == "dispram")
check("backend refuses anything else", raises(backend, "dumb", match="drm or dispram"))


class FakeDispram:
    """kindling's client: available() is the daemon answering; map_glued(nbytes, device) -> (pointer, the ordinary
    bytes, the reservation bytes lent above them)."""

    def __init__(self, up=True, lend=None):
        self.up, self.lend, self.calls = up, lend, []

    def available(self):
        return self.up

    def map_glued(self, nbytes, index):
        self.calls.append((nbytes, index))
        return 0x7000_0000_0000, nbytes - self.lend, self.lend


def via_dispram(client, call):
    """call() with ``client`` as the importable dispram module (None: not importable) and CUDA's device calls faked."""
    saved = sys.modules.get("dispram"), torch.cuda.current_device, torch.cuda.synchronize
    sys.modules["dispram"] = client
    torch.cuda.current_device, torch.cuda.synchronize = (lambda: 0), (lambda *a: None)
    try:
        return call()
    finally:
        if saved[0] is None:
            sys.modules.pop("dispram", None)
        else:
            sys.modules["dispram"] = saved[0]
        torch.cuda.current_device, torch.cuda.synchronize = saved[1], saved[2]


ordinary, display = 64 * MIB, 1792 * MIB
check("no client: refused", raises(lambda: via_dispram(None, lambda: dk._map_span_dispram(ordinary, display)),
                                   match="not importable"))
check("daemon down: refused", raises(lambda: via_dispram(FakeDispram(up=False, lend=display),
                                                         lambda: dk._map_span_dispram(ordinary, display)),
                                     match="does not answer"))
check("a slice short of the span: refused", raises(lambda: via_dispram(FakeDispram(lend=display - 16 * MIB),
                                                                       lambda: dk._map_span_dispram(ordinary, display)),
                                                   match="short of"))
client = FakeDispram(lend=display)
span = via_dispram(client, lambda: dk._map_span_dispram(ordinary, display))
check("the span is the client's one range", span.size == ordinary + display and client.calls == [(ordinary + display, 0)])
backend("dispram")
dk.connected_outputs = lambda sysfs, card: []                                            # noqa: E731
dk.SYSTEM = lambda: (_ for _ in ()).throw(AssertionError("the DRM path ran"))           # noqa: E731
client = FakeDispram(lend=display)
span = via_dispram(client, lambda: dk.map_span(ordinary, display))
check("map_span takes the dispram path, never DRM", span.size == ordinary + display and len(client.calls) == 1)
dk.connected_outputs = lambda sysfs, card: ["HDMI-A-1"]                                 # noqa: E731
check("a connected display is refused with dispram too",
      raises(lambda: via_dispram(FakeDispram(lend=display), lambda: dk.map_span(ordinary, display)),
             match="display connected"))
backend(None)

# -- the real span on this node ---------------------------------------------------------------------------------------
if "--gpu" in sys.argv:
    importlib.reload(dk)
    if GPU_BACKEND:
        os.environ[dk.BACKEND_ENV] = GPU_BACKEND
        print(f"---- the real span through {dk.backend()}", flush=True)
    os.environ[dk.ENV] = "1792"
    rows = 1 << 18                 # small planes, so the span takes many and a live server keeps its headroom
    planes = dk.latent_planes(17, rows, 512, "fp8", "cuda", dk.credit())
    k = len(planes)
    flat = [p.view(-1) for p in planes]
    total = k * flat[0].numel()

    def pattern(i, n):
        return (torch.arange(n, device="cuda", dtype=torch.int32) + 7 * i).remainder_(251).to(torch.uint8)

    for i, f in enumerate(flat):           # by a kernel, as the server writes them (a memcpy may not cross the halves)
        torch.bitwise_or(pattern(i, f.numel()), 0, out=f)
    torch.cuda.synchronize()
    same = all(torch.equal(f, pattern(i, f.numel())) for i, f in enumerate(flat))
    check(f"GPU writes read back across {k} planes ({total / MIB:.0f} MiB, top {dk.credit() // MIB} MiB display)", same)
    top = flat[-1][-MIB:].cpu()
    check("the display half's last MiB as written, read by the CPU", torch.equal(top, pattern(k - 1, flat[-1].numel())[-MIB:].cpu()))
    # a pool copy whose rows cross from the ordinary half into the display half (the plane holding the boundary)
    ordinary = total - dk.credit()
    j = ordinary // flat[0].numel()
    row_bytes = planes[j].shape[1]
    edge = (ordinary - j * flat[0].numel()) // row_bytes          # the row the display half starts in (or within)
    arena = Arena(rows, [Plane(planes[j], spanned=True)])
    before = torch.bitwise_or(planes[j], 0)                        # a copy by a kernel (clone() is a memcpy)
    arena.copy(edge - 300, edge + 1000, 600)                       # source across the boundary, destination above it
    arena.copy(edge + 2000, edge - 500, 700)                       # destination across it
    expect = before.clone()
    expect[edge + 1000:edge + 1600] = before[edge - 300:edge + 300]
    expect[edge - 500:edge + 200] = expect[edge + 2000:edge + 2700].clone()
    torch.cuda.synchronize()
    check(f"pool copies across the halves (plane {j}, boundary at row {edge})", torch.equal(planes[j], expect))
    try:
        planes[j][edge + 1000:edge + 1600].copy_(planes[j][edge - 300:edge + 300])
        torch.cuda.synchronize()
        print("     note: this driver now allows a memcpy across the halves")
    except Exception as exc:
        print(f"     a memcpy across the halves still fails ({str(exc).splitlines()[0][:50]}), as the kernel copy avoids")
    # bandwidth: a copy out of the display half against one out of ordinary device memory
    n = flat[-1].numel()
    src = flat[-1]
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    ref = torch.empty(n, dtype=torch.uint8, device="cuda")
    for name, a, b in (("display -> device", src, dst), ("device -> device", ref, dst)):
        for _ in range(3):
            b.copy_(a)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            b.copy_(a)
        e1.record()
        torch.cuda.synchronize()
        print(f"     {name}: {10 * n / (e0.elapsed_time(e1) / 1000) / 2 ** 30:.1f} GiB/s")

print(f"{len(failures)} failed" if failures else "all passed")
sys.exit(1 if failures else 0)
