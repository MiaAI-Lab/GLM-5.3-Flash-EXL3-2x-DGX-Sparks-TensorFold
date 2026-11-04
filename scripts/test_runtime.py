#!/usr/bin/env python3
"""Checks that keep the two runtimes honest.

Run from the repository root: python3 scripts/test_runtime.py

The scripts must not call docker or podman directly (scripts/runtime.sh is the one place that does), and the places
where the two runtimes genuinely differ must keep both branches. None of this needs a container runtime, a GPU or the
worker: it reads the scripts and runs the small functions of runtime.sh in a subshell.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["start.sh", "stop.sh", "start-tp3.sh", "scripts/config.sh", "scripts/nodes.sh",
           "scripts/prepare.sh", "scripts/publish-image.sh"]


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def code_lines(text):
    """The lines that are code: comments dropped, so prose about Docker does not trip the checks."""
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        yield line


def runtime_sh(script, env=None):
    """Source scripts/runtime.sh and run `script` in that shell (the functions, not a second shell)."""
    e = dict(os.environ, **(env or {}))
    return subprocess.run(["bash", "-c", "source scripts/runtime.sh; " + script],
                          cwd=ROOT, env=e, capture_output=True, text=True)


# the runtime's own subcommands: a line mentioning docker/podman followed by one of these is a direct call
SUBCOMMANDS = ("run|ps|inspect|logs|stop|rm|pull|push|build|save|load|tag|volume|image|info|top|exec|login|"
               "kill|start|create|history|diff|wait|attach")


class NoDirectRuntimeCalls(unittest.TestCase):
    def test_scripts_use_ctr(self):
        for name in SCRIPTS:
            for line in code_lines(read(name)):
                for verb in ("docker", "podman"):
                    # the runtime's own name in a value or a message is fine; one driving a subcommand is not
                    self.assertNotRegex(
                        line, rf"(?<![\w-]){verb}\s+({SUBCOMMANDS})\b",
                        f"{name}: calls {verb} directly: {line.strip()}")

    def test_worker_carries_the_runtime(self):
        nodes = read("scripts/nodes.sh")
        self.assertIn('"$(remote_ctr)$*"', nodes,
                      "worker() must prefix remote commands, or a one-liner says docker on a podman pair")

    def test_config_sources_runtime(self):
        cfg = read("scripts/config.sh")
        self.assertIn('source "$_cfg_root/scripts/runtime.sh"', cfg)
        self.assertLess(cfg.index("scripts/runtime.sh"), cfg.index('CONTAINER_NAME="${CONTAINER_NAME:-'))


class RuntimeHelpers(unittest.TestCase):
    def test_helpers_exist(self):
        text = read("scripts/runtime.sh")
        for fn in ("ctr()", "runtime_env()", "remote_ctr()", "rt_present()", "rt_require()",
                   "rt_save()", "rt_build_net()", "rt_root()", "rt_has_gpu()"):
            self.assertIn(fn, text, f"missing {fn}")

    def test_ctr_resolves_per_runtime(self):
        cases = [
            ({}, "docker"),                      # default
            ({"CONTAINER_RT": "podman"}, "podman"),
            ({"CONTAINER_RT": "podman", "CONTAINER_RT_SUDO": "1"}, "sudo -n podman"),
        ]
        for env, want in cases:
            out = runtime_sh('printf "%s\\n" "${RT[*]}"', env=env)
            self.assertEqual(out.stdout.strip(), want, f"RT for {env}")
        # ctr travels to the worker as a function body, so it must branch on the two variables itself (RT is not
        # there): it may not close over RT.
        body = read("scripts/runtime.sh")
        self.assertIn('sudo -n podman "$@"', body)
        self.assertIn('docker "$@"', body)

    def test_bad_runtime_is_refused(self):
        res = runtime_sh("true", env={"CONTAINER_RT": "lxc"})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("docker or podman", res.stderr + res.stdout)

    def test_sudo_flag_is_podman_only(self):
        res = runtime_sh("true", env={"CONTAINER_RT": "docker", "CONTAINER_RT_SUDO": "1"})
        self.assertNotEqual(res.returncode, 0)

    def test_build_net_only_for_podman(self):
        self.assertEqual(runtime_sh("rt_build_net").stdout.strip(), "")
        self.assertEqual(runtime_sh("rt_build_net", env={"CONTAINER_RT": "podman"}).stdout.strip(), "--network host")

    def test_save_format_only_for_podman(self):
        # dry: replace the runtime with an echo, then ask rt_save what it would run
        echo_ctr = 'source scripts/runtime.sh; ctr(){ echo "CTR $*"; }; rt_save img'
        self.assertIn("CTR save img", runtime_sh(echo_ctr).stdout)
        self.assertNotIn("docker-archive", runtime_sh(echo_ctr).stdout)
        self.assertIn("CTR save --format docker-archive img",
                      runtime_sh(echo_ctr, env={"CONTAINER_RT": "podman"}).stdout)

    def test_remote_env_is_exported(self):
        out = runtime_sh("runtime_env", env={"CONTAINER_RT": "podman", "CONTAINER_RT_SUDO": "1"})
        self.assertIn("export CONTAINER_RT=podman", out.stdout)
        self.assertIn("CONTAINER_RT_SUDO=1", out.stdout)


class RuntimeDifferences(unittest.TestCase):
    def test_shm_size_is_docker_only(self):
        start = read("start.sh")
        self.assertIn('[[ "$CONTAINER_RT" == podman ]] || RUN_ARGS+=(--shm-size 16g)', start,
                      "podman refuses --shm-size with --ipc=host")

    def test_cgroup_parent(self):
        start = read("start.sh")
        self.assertIn('RUN_ARGS+=(--cgroup-parent "$CGROUP_PARENT")', start)

    def test_nfs_volume_has_both_spellings(self):
        nodes = read("scripts/nodes.sh")
        self.assertIn('NFS_DEVICE="$server:$NFS_PATH"', nodes)
        self.assertIn('NFS_DEVICE=":$NFS_PATH"', nodes)
        self.assertIn('NFS_OPTS="addr=$server,', nodes)

    def test_image_root_per_runtime(self):
        rt = read("scripts/runtime.sh")
        self.assertIn("{{.Store.GraphRoot}}", rt)
        self.assertIn("{{.DockerRootDir}}", rt)
        self.assertIn("DOCKER_ROOT=$(rt_root)", read("scripts/prepare.sh"))

    def test_image_transfer_uses_docker_archive(self):
        self.assertIn("--format docker-archive", read("scripts/runtime.sh"))
        self.assertIn('rt_save "$IMAGE" | worker "$i" \'ctr load\'', read("scripts/prepare.sh"))

    def test_free_space_is_read_with_the_right_privilege(self):
        # rootful podman's store (/var/lib/containers, 0700 root) cannot be stat'ed or df'ed by the login user: the
        # free-space checks must go through rt_dev/rt_free_gb, which use sudo when the runtime does.
        prep = read("scripts/prepare.sh")
        self.assertNotIn("df -BG", prep, "prepare.sh must not df the store directly")
        self.assertNotIn('stat -c %d', prep, "prepare.sh must not stat the store directly")
        self.assertIn('rt_dev "$DOCKER_ROOT"', prep)
        self.assertIn('free_gb() { rt_free_gb "$1"; }', prep)
        rt = read("scripts/runtime.sh")
        self.assertIn("rt_needs_root()", rt)
        self.assertIn('sudo -n stat -c %d "$1"', rt)
        self.assertIn('sudo -n df -BG --output=avail "$1"', rt)

    def test_publish_logs_in_per_runtime(self):
        pub = read("scripts/publish-image.sh")
        self.assertIn("--authfile", pub)
        self.assertIn("DOCKER_CONFIG", pub)


if __name__ == "__main__":
    unittest.main(verbosity=2)
