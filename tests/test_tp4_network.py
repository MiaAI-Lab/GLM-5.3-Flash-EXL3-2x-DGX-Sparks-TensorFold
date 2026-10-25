"""Offline recipe contract checks. No Docker, SSH, GPU, or network is used.

Run BASH=/path/to/bash python3 -m unittest discover -s tests -p test_tp4_network.py -v.
Bash 4+ is required by the recipe's existing associative-array configuration.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "zig/TRANSPORT_SOURCE.json").read_text())["recipe_base_commit"]
BASH = os.environ.get("BASH", shutil.which("bash"))
if int(subprocess.check_output([BASH, "-c", "echo ${BASH_VERSINFO[0]}"], text=True).strip()) < 4:
    raise RuntimeError("These recipe tests require Bash 4+; set BASH=/path/to/bash")
KEYS = "TP ENGINE COMM SPLIT CONTEXT PARALLEL TF_GLM_SIDE TF_GLM_L2PF TF_ROCE_MAX_KB TF_GLM_MULTI_WINDOW KV_POOL_GIB MEMORY_RESERVE_GIB PROMPT_CACHE_GIB ZIG_HAS".split()
SELECTED = dict(TP4_NETWORK="registered-rdma", TP="4", ENGINE="zig", FABRIC_EXPECT="ring", SPLIT="0")


class NetworkProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tp4-network-offline-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "bin").mkdir()
        for name in ("config.sh", "nodes.sh", "memavail.sh"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        (self.root / "bin/bash").symlink_to(BASH)
        # BSD seq leaves a trailing separator with -s; production uses GNU seq.
        seq = self.root / "bin/seq"
        seq.write_text('#!/bin/sh\n"' + shutil.which("seq") + '" "$@" | sed "s/[[:space:]]*$//"\n')
        seq.chmod(0o755)
        self.calls = self.root / "calls"
        # Host-discovery stubs are deterministic; forbidden operations fail loudly.
        for name in ("hostname", "getent", "ssh", "docker", "nvidia-smi", "curl", "rsync"):
            content = '#!/bin/sh\nprintf "%s\\n" "' + name + '" >> "$CALL_LOG"\n'
            if name == "hostname":
                content += 'echo fixture-host\n'
            elif name == "getent":
                content += 'echo "192.0.2.9 STREAM fixture-host"\n'
            else:
                content += 'exit 91\n'
            path = self.root / "bin" / name
            path.write_text(content)
            path.chmod(0o755)
        # Clean environment: never forward developer credentials into fixtures.
        self.env = {"PATH": str(self.root / "bin") + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": str(self.root), "CALL_LOG": str(self.calls),
                    "WORKER": "fixture-rank1", "WORKER2": "fixture-rank2", "WORKER3": "fixture-rank3"}

    def run_shell(self, code, settings=None):
        return subprocess.run([BASH, "-c", "set -euo pipefail\n" + code], cwd=self.root,
                              env={**self.env, **(settings or {})}, text=True, capture_output=True, timeout=15)

    def config(self, settings=None):
        result = self.run_shell('source scripts/config.sh --validate-tp4-network\n' +
                                'for k in ' + ' '.join(KEYS + ["TP4_NETWORK", "TF_GLM_TP4_REGISTERED_RDMA"]) +
                                '; do printf "%s=%s\\n" "$k" "${!k-}"; done', settings)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return result, values

    def assert_no_host_calls(self):
        self.assertFalse(self.calls.exists(), self.calls.read_text() if self.calls.exists() else "")

    def test_opt_in_defaults_and_explicit_tuning(self):
        result, values = self.config(SELECTED)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({k: values[k] for k in ("COMM", "TF_GLM_TP4_REGISTERED_RDMA", "TF_ROCE_MAX_KB", "TF_GLM_SIDE", "TF_GLM_L2PF")},
                         dict(COMM="roce", TF_GLM_TP4_REGISTERED_RDMA="1", TF_ROCE_MAX_KB="512", TF_GLM_SIDE="1", TF_GLM_L2PF="1"))
        result, values = self.config({**SELECTED, "CONTEXT": "65536", "KV_POOL_GIB": "17", "TF_GLM_SIDE": "0", "TF_GLM_L2PF": "off"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([values[k] for k in ("CONTEXT", "KV_POOL_GIB", "TF_GLM_SIDE", "TF_GLM_L2PF")], ["65536", "17", "0", "off"])

    def test_invalid_profiles_fail_before_discovery(self):
        bad = [{"TP4_NETWORK": "unknown"}, {"TP4_NETWORK": ""},
               {"TF_GLM_TP4_REGISTERED_RDMA": "1"}]
        bad += [{**SELECTED, **change} for change in (
            {"ENGINE": "python"}, {"TP": "2"}, {"TP": "3"}, {"FABRIC_EXPECT": "switch"},
            {"SPLIT": "1"}, {"COMM": "nccl"}, {"COMM": "bogus"},
            {"TF_GLM_TP4_REGISTERED_RDMA": "0"}, {"TF_GLM_ROCE": "0"},
            {"TF_ROCE_RING": "0"}, {"TF_ROCE_RING": "bad"}, {"TF_ROCE_RING": ""}, {"TF_ROCE_MAX_KB": "1024"}, {"TF_ROCE_MAX_KB": ""})]
        for settings in bad:
            with self.subTest(settings=settings):
                result, _ = self.config(settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TP4_NETWORK", result.stderr)
                self.assert_no_host_calls()

    def test_local_file_and_environment_precedence(self):
        (self.root / "scripts/local.sh").write_text('TP4_NETWORK=registered-rdma\nTP=4\nFABRIC_EXPECT=ring\nSPLIT=0\nTF_GLM_SIDE=0\n')
        (self.root / ".env").write_text('TP4_NETWORK=upstream\nTF_GLM_SIDE=1\n')
        result, values = self.config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["TP4_NETWORK"], "registered-rdma")
        self.assertEqual(values["TF_GLM_SIDE"], "0")
        result, values = self.config(dict(TP4_NETWORK="upstream", COMM="nccl", TF_GLM_SIDE="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["TF_GLM_TP4_REGISTERED_RDMA"], "0")
        self.assertEqual(values["TF_GLM_SIDE"], "1")
        (self.root / "scripts/local.sh").unlink()
        (self.root / ".env").write_text('\n'.join(k + '=' + v for k, v in SELECTED.items()) + '\n')
        result, values = self.config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["TF_GLM_TP4_REGISTERED_RDMA"], "1")

    def test_upstream_config_matches_pinned_recipe(self):
        baseline = subprocess.check_output(["git", "show", BASELINE + ":scripts/config.sh"], cwd=ROOT, text=True)
        current = (self.root / "scripts/config.sh").read_text()
        for settings in ({}, dict(TP="3", SPLIT="0"), dict(TP="4", FABRIC_EXPECT="ring", SPLIT="0", COMM="nccl"),
                         dict(ENGINE="python"), dict(ENGINE="python", TP="4", FABRIC_EXPECT="ring")):
            with self.subTest(settings=settings):
                (self.root / "scripts/config.sh").write_text(current)
                result, actual = self.config(settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                (self.root / "scripts/config.sh").write_text(baseline)
                result, expected = self.config(settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({k: actual[k] for k in KEYS}, {k: expected[k] for k in KEYS})
                self.assertEqual(actual["TF_GLM_TP4_REGISTERED_RDMA"], "0")
        (self.root / "scripts/config.sh").write_text(current)

    def zig_fixture(self):
        zig = self.root / "zig"
        zig.mkdir()
        for name in ("patches", "aot", "Dockerfile"):
            (zig / name).symlink_to(ROOT / "zig" / name)
        (zig / "transports").mkdir()
        shutil.copyfile(ROOT / "zig/transports/0024-zig-glm-tp4-registered-relay.patch", zig / "transports/0024-zig-glm-tp4-registered-relay.patch")

    def test_upstream_image_hash_and_build_inputs_match_pinned_recipe(self):
        self.zig_fixture()
        # First prove the fixture's shared inputs really are the upstream bytes.
        result = subprocess.run(["git", "diff", "--exit-code", BASELINE, "--",
                                 "zig/patches", "zig/aot", "zig/Dockerfile"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        current = (self.root / "scripts/config.sh").read_text()
        actual = self.run_shell('source scripts/config.sh\nimage_hash')
        baseline = subprocess.check_output(["git", "show", BASELINE + ":scripts/config.sh"], cwd=ROOT, text=True)
        (self.root / "scripts/config.sh").write_text(baseline)
        expected = self.run_shell('source scripts/config.sh\nimage_hash')
        (self.root / "scripts/config.sh").write_text(current)
        self.assertEqual(actual.returncode, 0, actual.stderr)
        self.assertEqual(expected.returncode, 0, expected.stderr)
        self.assertEqual(actual.stdout, expected.stdout)
        staged = self.run_shell('source scripts/config.sh\nzig_stage_patches staged')
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / "staged").iterdir()},
                         {p.name: p.read_bytes() for p in (ROOT / "zig/patches").glob("*.patch")})

    def test_only_selected_image_stages_and_hashes_the_transport_patch(self):
        self.zig_fixture()
        code = 'source scripts/config.sh\nimage_hash'
        upstream = self.run_shell(code)
        selected = self.run_shell(code, SELECTED)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertNotEqual(upstream.stdout, selected.stdout)
        optional = self.root / "zig/transports/0024-zig-glm-tp4-registered-relay.patch"
        optional.write_bytes(optional.read_bytes() + b'\n')
        self.assertEqual(self.run_shell(code).stdout, upstream.stdout)
        self.assertNotEqual(self.run_shell(code, SELECTED).stdout, selected.stdout)
        staged = self.run_shell('source scripts/config.sh\nzig_stage_patches staged', SELECTED)
        self.assertEqual(staged.returncode, 0, staged.stderr)
        names = sorted(p.name for p in (self.root / "staged").iterdir())
        self.assertEqual(names[-1], optional.name)
        self.assertEqual(len(names), len(list((ROOT / "zig/patches").glob("*.patch"))) + 1)
        self.assertEqual((self.root / "staged" / optional.name).read_bytes(), optional.read_bytes())
        optional.unlink()
        failed = self.run_shell('source scripts/config.sh\nzig_stage_patches missing', SELECTED)
        self.assertNotEqual(failed.returncode, 0)
        self.assertLessEqual(set(self.calls.read_text().splitlines()), {"hostname", "getent"})

    def test_selected_image_labels_are_required_on_every_rank_before_stop(self):
        code = '''source scripts/config.sh --validate-tp4-network
source scripts/nodes.sh
image_hash() { echo 123456789abc; }
docker() {
  echo "rank0 $*" >> "$IMAGE_CALLS"
  [[ "${MISSING_RANK-}" != 0 ]] || return 91
  echo "$LABEL_0"
}
worker() {
  echo "rank$1 $*" >> "$IMAGE_CALLS"
  [[ "${MISSING_RANK-}" != "$1" ]] || return 91
  local label="LABEL_$1"
  echo "${!label}"
}
require_tp4_image_profile
echo PASSED_IMAGE_BOUNDARY
'''
        settings = {**SELECTED, "IMAGE": "fixture/override:selected", "PREPARE": "0",
                    "IMAGE_CALLS": str(self.root / "image-calls"),
                    **{f"LABEL_{rank}": "123456789abc" for rank in range(4)}}
        result = self.run_shell(code, settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASSED_IMAGE_BOUNDARY", result.stdout)
        calls = Path(settings["IMAGE_CALLS"])
        self.assertEqual(len(calls.read_text().splitlines()), 4)
        self.assertEqual(calls.read_text().count("fixture/override:selected"), 4)
        for rank in range(4):
            for change in ({f"LABEL_{rank}": "upstreamhash"}, {f"LABEL_{rank}": ""}, {"MISSING_RANK": str(rank)}):
                with self.subTest(rank=rank, change=change):
                    result = self.run_shell(code, {**settings, **change})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("PASSED_IMAGE_BOUNDARY", result.stdout)
                    self.assertIn("TP4_NETWORK=registered-rdma", result.stderr)
        calls.unlink()
        result = self.run_shell(code, {**settings, "TP4_NETWORK": "upstream", "COMM": "nccl"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(calls.exists())
        start = (ROOT / "start.sh").read_text()
        self.assertLess(start.index("\nrequire_tp4_image_profile\n"), start.index("\n  ./stop.sh\n"))

    def test_start_guard_preserves_default_and_accepts_opt_in(self):
        # Run the real start preflight and stop immediately before the banner and
        # all Docker/preparation operations. Repository source remains untouched.
        source = (ROOT / "start.sh").read_text()
        marker = "# ---------------------------------------------------------------- banner and progress"
        source = source.replace(marker, 'echo OFFLINE_PREFLIGHT_PASSED\nexit 0\n' + marker, 1)
        (self.root / "start.sh").write_text(source)
        cases = [(dict(TP="4", FABRIC_EXPECT="ring", SPLIT="0", COMM="nccl"), True),
                 (dict(TP="4", FABRIC_EXPECT="ring", SPLIT="0", COMM="roce"), False),
                 (SELECTED, True), ({**SELECTED, "COMM": "nccl"}, False)]
        for settings, valid in cases:
            with self.subTest(settings=settings):
                result = self.run_shell('exec "$TEST_BASH" ./start.sh restart', {**settings, "TEST_BASH": BASH})
                self.assertEqual(result.returncode == 0, valid, result.stderr)
                self.assertEqual("OFFLINE_PREFLIGHT_PASSED" in result.stdout, valid)
                if valid and settings.get("TP4_NETWORK") == "registered-rdma":
                    self.assertIn("known-working NCCL", result.stderr)
                    self.assertIn("separately patched NCCL 2.30.7", result.stderr)
                    self.assertIn("version alone", result.stderr)
                elif valid:
                    self.assertNotIn("known-working NCCL", result.stderr)
        self.assertLessEqual(set(self.calls.read_text().splitlines()), {"hostname", "getent"})

    def test_prepare_rejects_invalid_profile_before_host_calls(self):
        shutil.copyfile(ROOT / "scripts/prepare.sh", self.root / "scripts/prepare.sh")
        result = self.run_shell('exec "$TEST_BASH" scripts/prepare.sh',
                                {**SELECTED, "COMM": "nccl", "TEST_BASH": BASH})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TP4_NETWORK", result.stderr)
        self.assert_no_host_calls()

    def test_stop_remains_usable_with_stale_or_missing_profile_settings(self):
        # Execute the complete real stop.sh in dry-run mode, using local stubs for
        # container presence and all worker reads; no SSH or Docker is executed.
        shutil.copyfile(ROOT / "stop.sh", self.root / "stop.sh")
        # stop.sh targets GNU userland, where paste accepts omitted stdin '-'.
        # Supply it explicitly in the fixture so this test also runs on BSD/macOS.
        paste = self.root / "bin/paste"
        paste.write_text('#!/bin/sh\nexec ' + shutil.which("paste") + ' "$@" -\n')
        paste.chmod(0o755)
        (self.root / "bin/docker").write_text(
            '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$CALL_LOG"\n'
            'case "$1" in ps) echo glm53-flash-tf ;; inspect) echo false ;; *) exit 97 ;; esac\n')
        (self.root / "bin/ssh").write_text(
            '#!/bin/sh\nprintf "ssh %s\\n" "$*" >> "$CALL_LOG"\nexit 0\n')
        for settings in (dict(TP4_NETWORK="registered-rdma"),
                         dict(TP4_NETWORK="misspelled", TF_GLM_TP4_REGISTERED_RDMA="1"),
                         {**SELECTED, "TF_ROCE_RING": "0"},
                         {**SELECTED, "COMM": "nccl", "TF_ROCE_MAX_KB": "1024"}):
            with self.subTest(settings=settings):
                result = self.run_shell('exec "$TEST_BASH" ./stop.sh',
                                        {**settings, "TEST_BASH": BASH, "DRY_RUN": "1"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("would stop and remove", result.stdout)
                self.assertIn("rank 0 here", result.stdout)
                self.assertIn("rank 3 on fixture-rank3", result.stdout)
                self.assertNotIn("config.sh: TP4_NETWORK", result.stderr)
        calls = self.calls.read_text()
        self.assertIn("docker ps", calls)
        self.assertIn("fixture-rank3", calls)
        self.assertNotIn("docker stop", calls)
        self.assertNotIn("docker rm", calls)

    def fixture_inventories(self, fabric, hcas=4, uniform=False, order=(0, 1, 2, 3)):
        """Inventories are synthetic IPv4 documentation ranges, with per-rank NIC names.

        Ring port order deliberately crosses per-rank HCA indices (XOR layout).
        """
        for rank in range(4):
            lines = [f"default fixture-net-{rank}"]
            for port in range(hcas):
                if fabric == "mesh":
                    subnet = 10 if uniform else 10 + port
                    address = 1 + rank + (16 * port if uniform else 0)
                else:
                    pos = order.index(rank)
                    edge = (pos - 1) % 4 if (port % 2) == (rank % 2) else pos
                    subnet = 30 + edge
                    address = 1 + rank + 16 * (port // 2)
                lines.append(f"nic-{rank}-{port} hca-{rank}-{port} 192.0.{subnet}.{address}/24 {rank + 3}")
            (self.root / f"inventory-{rank}").write_text("\n".join(lines) + "\n")

    def fixture_launcher(self):
        """Run real wrappers/preflight/discovery/forwarding, replacing system reads.

        The launcher exits before its preparation, Docker and stop operations.
        graph_xml is real; graph placement is redirected to temporary files.
        """
        source = (ROOT / "start.sh").read_text()
        functions = source[source.index("zig_env() {"):source.index("\nenv_args\nRUN_ARGS=")]
        fixture = r'''
node_inventory() { cat inventory-0; }
worker() { cat "inventory-$1"; }
link_info() { echo "192.0.2.9 fixture-net-0 fixture-hca-0 3"; }
ring_graphs() {
  local r
  for r in 0 1 2 3; do
    NODE_GRAPH[r]="$PWD/graph-$r"
    graph_xml "$r" "$r" "$(( (r + 3) % 4 ))" > "${NODE_GRAPH[r]}.xml"
  done
}
# The stop sentinel must never be reached after failed topology validation.
[[ "${FIXTURE_UNREACHABLE:-0}" != 1 ]] || WORKER_DOWN[2]=1
detect_links
echo DISCOVERY_PASSED_BEFORE_STOP
echo "FABRIC=$FABRIC"
'''
        fixture += functions + r'''
env_args
for rank in 0 1 2 3; do
  echo "RANK=$rank"
  printf '%s\n' "${ENV_ARGS[@]}"
  rank_nccl_env "$rank"
done
exit 0
'''
        marker = "# ---------------------------------------------------------------- banner and progress"
        self.assertIn(marker, source)
        (self.root / "start.sh").write_text(source.replace(marker, fixture + "\n" + marker, 1))
        (self.root / "start.sh").chmod(0o755)
        for wrapper in ("start-tp4.sh", "start-tp4-switchless.sh"):
            shutil.copyfile(ROOT / wrapper, self.root / wrapper)

    def test_ring_wrapper_forwards_two_or_four_hcas(self):
        self.fixture_launcher()
        for hcas in (2, 4):
            with self.subTest(hcas=hcas):
                self.fixture_inventories("ring", hcas)
                settings = {**SELECTED, "TF_GLM_SIDE": "0", "TF_GLM_L2PF": "off",
                            "KV_POOL_GIB": "17", "CONTEXT": "65536", "TF_ROCE_HCA": "stale-global-hca"}
                result = self.run_shell('exec "$TEST_BASH" ./start-tp4-switchless.sh restart', {**settings, "TEST_BASH": BASH})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("FABRIC=ring", result.stdout)
                self.assertNotIn("stale-global-hca", result.stdout)
                self.assertIn("known-working NCCL", result.stderr)
                self.assertEqual(result.stdout.count("TF_ROCE_RING=1"), 4)
                self.assertNotIn("TF_ROCE_RING=0", result.stdout)
                for flag in ("TF_GLM_TP4_REGISTERED_RDMA=1", "TF_GLM_ROCE=1", "TF_GLM_SIDE=0", "TF_GLM_L2PF=0", "TF_GLM_CACHE_GIB=17", "TF_ROCE_MAX_KB=512"):
                    self.assertEqual(result.stdout.splitlines().count(flag), 4, flag)
                for rank in range(4):
                    rails = ",".join(f"hca-{rank}-{port}" for port in range(hcas))
                    self.assertIn(f"TF_ROCE_HCA={rails}", result.stdout)
                    self.assertIn(f"NCCL_IB_GID_INDEX={rank + 3}", result.stdout)
                    graph = (self.root / f"graph-{rank}.xml").read_text()
                    self.assertNotIn('net dev="-1"', graph)
                    self.assertIn("NCCL_GRAPH_FILE=/nccl-graph.xml", result.stdout)
        self.assertLessEqual(set(self.calls.read_text().splitlines()), {"hostname", "getent"})

    def test_discovered_mesh_is_refused_before_stop(self):
        self.fixture_launcher()
        settings = {k: v for k, v in SELECTED.items() if k != "FABRIC_EXPECT"}
        for fabric in ("mesh", "ring"):
            self.fixture_inventories(fabric)
            for requested in (None, "0", "1"):
                with self.subTest(fabric=fabric, requested=requested):
                    tuned = {**settings, "TEST_BASH": BASH}
                    if requested is not None:
                        tuned["TF_ROCE_RING"] = requested
                    result = self.run_shell('exec "$TEST_BASH" ./start.sh restart', tuned)
                    valid = fabric == "ring" and requested != "0"
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    self.assertEqual("DISCOVERY_PASSED_BEFORE_STOP" in result.stdout, valid)
                    if not valid:
                        self.assertIn("physical ring", result.stderr)
                    else:
                        self.assertEqual(result.stdout.count("TF_ROCE_RING=1"), 4)
        self.assertLessEqual(set(self.calls.read_text().splitlines()), {"hostname", "getent"})

    def test_wrong_wrapper_missing_peer_and_wrong_ring_order_fail_closed(self):
        self.fixture_launcher()
        for fabric, wrapper in (("mesh", "start-tp4-switchless.sh"), ("ring", "start-tp4.sh")):
            self.fixture_inventories(fabric)
            result = self.run_shell(f'exec "$TEST_BASH" ./{wrapper} restart', {**SELECTED, "TEST_BASH": BASH})
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("DISCOVERY_PASSED_BEFORE_STOP", result.stdout)
            self.assertIn("switch", result.stderr)
        for fabric in ("mesh", "ring"):
            self.fixture_inventories(fabric)
            (self.root / "inventory-2").write_text("default fixture-net-2\n")
            result = self.run_shell('exec "$TEST_BASH" ./start.sh restart', {**SELECTED, "FABRIC_EXPECT": "", "TEST_BASH": BASH})
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("DISCOVERY_PASSED_BEFORE_STOP", result.stdout)
        self.fixture_inventories("ring", order=(0, 2, 1, 3))
        result = self.run_shell('exec "$TEST_BASH" ./start.sh restart', {**SELECTED, "TEST_BASH": BASH})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("workers are not in its order", result.stderr)
        self.assertNotIn("DISCOVERY_PASSED_BEFORE_STOP", result.stdout)
        # The real start discovers topology before its first stop call.
        start = (ROOT / "start.sh").read_text()
        self.assertLess(start.index("\ndetect_links\n"), start.index("\n  ./stop.sh\n"))

    def test_explicit_fabric_ring_mismatch_fails_before_all_host_calls(self):
        for fabric, ring in (("mesh", "1"), ("ring", "0")):
            result, _ = self.config({**SELECTED, "FABRIC_EXPECT": fabric, "TF_ROCE_RING": ring})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("physical ring", result.stderr)
            self.assert_no_host_calls()

    def test_start_restart_and_prepare_reject_static_conflicts_before_host_calls(self):
        shutil.copyfile(ROOT / "start.sh", self.root / "start.sh")
        shutil.copyfile(ROOT / "scripts/prepare.sh", self.root / "scripts/prepare.sh")
        for entry in ("./start.sh", "./start.sh restart", "scripts/prepare.sh"):
            for changes in ({"COMM": "nccl"}, {"FABRIC_EXPECT": "mesh", "TF_ROCE_RING": "1"},
                            {"TP4_NETWORK": "unknown"}, {"TF_GLM_TP4_REGISTERED_RDMA": "0"},
                            {"TF_ROCE_MAX_KB": "1024"}):
                with self.subTest(entry=entry, changes=changes):
                    result = self.run_shell(f'exec "$TEST_BASH" {entry}', {**SELECTED, **changes, "TEST_BASH": BASH})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("TP4_NETWORK", result.stderr)
                    self.assert_no_host_calls()

    def test_unreachable_dry_run_inventory_cannot_qualify_opt_in(self):
        self.fixture_launcher()
        self.fixture_inventories("ring")
        result = self.run_shell('exec "$TEST_BASH" ./start-tp4-switchless.sh restart',
                                {**SELECTED, "TEST_BASH": BASH, "DRY_RUN": "1", "FIXTURE_UNREACHABLE": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires every worker inventory", result.stderr)
        self.assertNotIn("DISCOVERY_PASSED_BEFORE_STOP", result.stdout)

    def test_cutoff_growth_is_skipped_only_for_opt_in(self):
        for selected, expected in ((True, "512"), (False, "1024")):
            settings = {**SELECTED, "TP4_NETWORK": "registered-rdma" if selected else "upstream", "TF_GLM_MULTI_WINDOW": "64"}
            result, values = self.config(settings)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(values["TF_ROCE_MAX_KB"], expected)

    def test_actual_forwarding_functions_keep_all_rank_flags_and_local_rails(self):
        source = (ROOT / "start.sh").read_text()
        functions = source[source.index("zig_env() {"):source.index("\nenv_args\nRUN_ARGS=")]
        code = 'source scripts/config.sh\nsource scripts/nodes.sh\n' + functions + '''
env_args
printf '%s\\n' "${ENV_ARGS[@]}"
FABRIC=ring
for rank in 0 1 2 3; do
  NODE_DEV[$rank]="fixture-net-$rank"
  NODE_HCAS[$rank]="fixture-hca-$rank-a,fixture-hca-$rank-b"
  NODE_GID[$rank]=$((rank + 4))
  rank_nccl_env "$rank"
done
'''
        result = self.run_shell(code, {**SELECTED, "TF_GLM_SIDE": "0", "TF_GLM_L2PF": "off", "TF_ROCE_HCA": "must-not-be-forwarded"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("TF_GLM_TP4_REGISTERED_RDMA=1", "TF_GLM_ROCE=1", "TF_GLM_L2PF=0", "TF_GLM_SIDE=0", "TF_ROCE_MAX_KB=512"):
            self.assertIn(flag, result.stdout.splitlines())
        self.assertNotIn("must-not-be-forwarded", result.stdout)
        for rank in range(4):
            self.assertIn(f"TF_ROCE_HCA=fixture-hca-{rank}-a,fixture-hca-{rank}-b", result.stdout)
            self.assertIn(f"NCCL_IB_GID_INDEX={rank + 4}", result.stdout)
        self.assertEqual(result.stdout.count("TF_ROCE_RING=1"), 4)
        self.assertLessEqual(set(self.calls.read_text().splitlines()), {"hostname", "getent"})


if __name__ == "__main__":
    unittest.main()
