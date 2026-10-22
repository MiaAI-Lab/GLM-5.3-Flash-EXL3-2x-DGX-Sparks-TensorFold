"""Run inside the build stage before compiling anything. Exact Git blob bytes."""
import hashlib
import json
import os
from pathlib import Path


def require(ok, message):
    if not ok:
        raise SystemExit(message)


root = Path('/tensorfold')
manifest = Path('/tmp/native-source-blobs-be94e91.json')
expected = json.loads(manifest.read_text())
seen = set()
for folder, dirs, files in os.walk(root):
    if Path(folder) == root:
        dirs[:] = [d for d in dirs if d != '.git']
    for name in files:
        path = Path(folder) / name
        relative = str(path.relative_to(root))
        seen.add(relative)
        data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        require(expected.get(relative) == blob, 'Focused source blob mismatch: ' + relative)
require(seen == set(expected), 'Focused source file inventory differs')
require(not (root / '.zig-cache').exists() and not (root / 'zig-out').exists(), 'Expected clean source before real compilation')
require(not Path('/opt/tensorfold/fatbin').exists(), 'Preexisting fatbins refused')
require(Path('/usr/include/infiniband/verbs.h').is_file(), 'Real verbs headers required')
print(json.dumps({'native_commit': 'be94e919e35c628261bd38364b1743a2e4353a26',
                  'focused_source_bytes_passed': True, 'files': len(seen),
                  'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'real_verbs_headers': True, 'clean_source_before_compile': True}))
