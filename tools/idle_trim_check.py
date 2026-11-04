"""Patch 0130's check on a GPU, in an image that has the patch (about 25 s, needs ~12 GiB of free GPU memory):
  docker run --rm --gpus all -v "$PWD/tools:/t:ro" --entrypoint python <image> /t/idle_trim_check.py
It runs the real trim_memory and MultiDecoder._trim_idle on a stand-in decoder: no trim while busy, one trim per idle
spell, torch's cached 8 GiB back to the system (MemAvailable rises), a tensor held across the trim untouched.
"""
import threading, time, torch
from tensorfold.families.glm5_next.cuda import multi

def avail_gib():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 2**20

assert multi.idle_trim_seconds("") == 0 and multi.idle_trim_seconds("60") == 60
for bad in ("-1", "x"):
    try: multi.idle_trim_seconds(bad); raise SystemExit("accepted " + bad)
    except ValueError: pass

class Stub: pass
d = Stub(); d.idle_trim_s = 2.0; d.idle = False; d.rank = 0; d.trims = 0; d.trimmed_bytes = 0
d.trim_stop = threading.Event(); d.device_index = torch.cuda.current_device()
t = threading.Thread(target=lambda: multi.MultiDecoder._trim_idle(d), daemon=True); t.start()

# busy: scratch cached by torch (what a long prefill leaves), kept tensor still held
keep = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
x = torch.empty(8 << 30, dtype=torch.uint8, device="cuda"); del x
res0, a0 = torch.cuda.memory_reserved() / 2**30, avail_gib()
print(f"busy:  reserved {res0:.2f} GiB, allocated {torch.cuda.memory_allocated()/2**30:.2f}, MemAvailable {a0:.1f}")
time.sleep(7)
assert d.trims == 0 and torch.cuda.memory_reserved() / 2**30 >= res0 - 0.01, "trimmed while busy"
d.idle = True
time.sleep(5)  # 2 s idle + poll
res1, a1 = torch.cuda.memory_reserved() / 2**30, avail_gib()
print(f"idle:  reserved {res1:.2f} GiB, MemAvailable {a1:.1f}, trims {d.trims}, trimmed {d.trimmed_bytes/2**30:.2f} GiB")
assert d.trims == 1 and res0 - res1 > 7.5, "idle trim did not free the cached scratch"
assert a1 - a0 > 7.0, "MemAvailable did not rise"
assert keep.numel() == 2 << 30 and keep[0].item() == keep[0].item()   # held tensor untouched
time.sleep(5); assert d.trims == 1, "trimmed twice in one idle spell"
d.idle = False; time.sleep(6)
x = torch.empty(4 << 30, dtype=torch.uint8, device="cuda"); del x; d.idle = True; time.sleep(5)
assert d.trims == 2, "no trim in the second idle spell"
d.trim_stop.set()
print("OK")
