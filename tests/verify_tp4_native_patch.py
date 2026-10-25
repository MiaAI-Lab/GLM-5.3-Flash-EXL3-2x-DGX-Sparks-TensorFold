#!/usr/bin/env python3
"""Apply the complete recipe to pinned local source and compare frozen native bytes.

No network or GPU use. --upstream-repo must contain the pinned TensorFold commit;
--native-repo must contain the focused native commit in TRANSPORT_SOURCE.json.
The optional --zig path runs the native host contract checks on the reconstruction.
Existing Git checkouts and the recipe are read-only; reconstruction is temporary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(repo, commit, destination):
    destination.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], stdout=stream, check=True)
        stream.seek(0)
        subprocess.run(["tar", "-x", "-C", str(destination)], stdin=stream, check=True)


def inventory(root):
    out = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in source archive: {path}")
        if path.is_file():
            out[str(path.relative_to(root))] = digest(path)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--zig", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "zig/TRANSPORT_SOURCE.json").read_text())
    for name, expected in {**manifest["prior_patches_sha256"], manifest["patch"]: manifest["patch_sha256"]}.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"patch checksum mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="glm-focused-patch-verify-") as temp:
        work = Path(temp) / "applied"
        frozen = Path(temp) / "frozen"
        archive(args.upstream_repo, manifest["tensorfold_base_commit"], work)
        archive(args.native_repo, manifest["focused_native_commit"], frozen)
        patches = [*sorted((ROOT / "zig/patches").glob("*.patch")), ROOT / manifest["patch"]]
        for patch in patches:
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=work, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=work, check=True)
        actual, expected = inventory(work), inventory(frozen)
        if actual != expected:
            names = set(actual) | set(expected)
            raise RuntimeError("source byte mismatch: " + ", ".join(sorted(n for n in names if actual.get(n) != expected.get(n))))
        mode_differences = {}
        for name in actual:
            a = bool((work / name).stat().st_mode & stat.S_IXUSR)
            b = bool((frozen / name).stat().st_mode & stat.S_IXUSR)
            if a != b:
                mode_differences[name] = {"frozen_native": "100755" if b else "100644", "recipe_reconstruction": "100755" if a else "100644"}
        if mode_differences != manifest["known_reconstruction_mode_differences"]:
            raise RuntimeError("unexpected executable-mode differences: " + json.dumps(mode_differences, sort_keys=True))
        for name, expected_hash in manifest["result_file_sha256"].items():
            if actual.get(name) != expected_hash:
                raise RuntimeError(f"focused file digest mismatch: {name}")
        if args.zig:
            subprocess.run([str(args.zig), "build", "test-glm-transport-host", "test-glm-transport-probe-host",
                            "glm-transport-probe-host-build", "--summary", "all"], cwd=work, check=True)
        inventory_hash = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        print(json.dumps({"result": "PASS", "patch_sha256": manifest["patch_sha256"], "file_count": len(actual),
                          "all_source_bytes_equal": True, "file_sha256_inventory_digest": inventory_hash,
                          "document_omission": None, "known_mode_differences": mode_differences,
                          "host_checks_run": bool(args.zig), "gpu_checks_run": False}, indent=2))


if __name__ == "__main__":
    main()
