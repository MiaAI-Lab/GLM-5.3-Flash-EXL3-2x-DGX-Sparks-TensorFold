"""Export the exact real-NVCC fatbin inventory after check_fatbins validates SM121 SASS."""
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise SystemExit(message)


root = Path('/opt/tensorfold/fatbin')
expected = json.loads(Path('/tmp/expected-fatbins-be94e91.json').read_text())
require(len(expected) == len(set(expected)) == 56, 'Invalid expected fatbin set')
files = sorted(root.iterdir())
require({p.name for p in files} == set(expected), 'Real fatbin inventory differs (missing or extra files)')
require(all(p.is_file() and not p.is_symlink() and p.stat().st_size > 0 for p in files), 'Invalid/empty real fatbin')
record = {'native_commit': 'be94e919e35c628261bd38364b1743a2e4353a26', 'sm': 121,
          'count': len(files), 'sass_checked_by': 'zig/build/check_fatbins.zig',
          'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
destination = Path('/opt/tensorfold/share/doc/focused-fatbins.json')
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps({'real_nvcc_fatbins': 56, 'exact_inventory': True, 'sm': 121}))
