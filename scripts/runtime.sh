# The container runtime: Docker (the default) or Podman, chosen with CONTAINER_RT. Sourced by config.sh, so every
# script in this recipe (start.sh, stop.sh, scripts/prepare.sh, scripts/port.sh) gets it, and by scripts/memguard.sh,
# which is copied to the worker on its own.
#
#   CONTAINER_RT=docker        (default) the docker CLI and daemon
#   CONTAINER_RT=podman        the podman CLI; the calls are the same
#   CONTAINER_RT_SUDO=1        run podman as root ("sudo -n podman"): rootful podman. It is what this recipe needs on
#                              a Spark pair: the worker's read-only NFS volume (--driver local --opt type=nfs) can only
#                              be mounted by root, and a system cgroup parent (CGROUP_PARENT, below) is root's.
#   CONTAINER_RT_SUDO=0        (default) podman as the login user (rootless); fine for the image-only helpers, not for
#                              the NFS volume or a Spark's system slice
#   CGROUP_PARENT=<slice>      optional, podman only: --cgroup-parent for each rank's container. Set it to the slice
#                              the host tuning made (e.g. tensorfold.slice with MemorySwapMax=0) so a rank cannot swap.
#
# ctr is the one command word the scripts use: it resolves to docker, podman or "sudo -n podman" at call time, so a
# function sent to the worker (save_log, spill_fix_container) behaves the same there once CONTAINER_RT and
# CONTAINER_RT_SUDO are in its environment (runtime_env prints that prefix).
CONTAINER_RT="${CONTAINER_RT:-docker}"
CONTAINER_RT_SUDO="${CONTAINER_RT_SUDO:-0}"
case "$CONTAINER_RT" in
  docker)
    RT=(docker); RT_WORDS=docker
    [[ "$CONTAINER_RT_SUDO" == 0 ]] || { echo "ERROR: CONTAINER_RT_SUDO is for podman; docker talks to its daemon" >&2; exit 1; }
    ;;
  podman)
    if [[ "$CONTAINER_RT_SUDO" == 1 ]]; then RT=(sudo -n podman); RT_WORDS='sudo -n podman'; else RT=(podman); RT_WORDS=podman; fi
    ;;
  *)
    echo "ERROR: CONTAINER_RT is docker or podman, not $CONTAINER_RT" >&2; exit 1 ;;
esac

# ctr: the runtime with its arguments (identical flags under both: run, inspect -f '{{...}}', ps --format, volume
# create --opt, image inspect, save, load, logs, kill, stop, rm, build). Read at call time: a function sent to the
# worker needs only the two variables.
ctr() {
  if [[ "${CONTAINER_RT:-docker}" == podman ]]; then
    if [[ "${CONTAINER_RT_SUDO:-0}" == 1 ]]; then sudo -n podman "$@"; else podman "$@"; fi
  else
    docker "$@"
  fi
}

# runtime_env: CONTAINER_RT and CONTAINER_RT_SUDO as a shell prefix for a command sent to the worker over ssh
# (worker 1 "$(runtime_env)$(declare -f ctr); ctr ps ..."). export, not a bare assignment: a plain "VAR=x; cmd" is a
# shell variable, which the guard that memguard.sh re-execs through setsid would not inherit.
runtime_env() { printf 'export CONTAINER_RT=%q CONTAINER_RT_SUDO=%q; ' "$CONTAINER_RT" "$CONTAINER_RT_SUDO"; }

# rt_present: the runtime's CLI is installed (start.sh's check; it does not talk to the daemon).
rt_present() { command -v "$CONTAINER_RT" >/dev/null || die "$CONTAINER_RT is not installed"; }

# rt_require: the runtime is installed and answers. $1 is the script's own wording for what it was doing.
rt_require() {
  command -v "$CONTAINER_RT" >/dev/null || die "$CONTAINER_RT is not installed"
  ctr info >/dev/null 2>&1 || {
    if [[ "$CONTAINER_RT" == podman && "$CONTAINER_RT_SUDO" == 1 ]]; then
      die "cannot talk to rootful podman (sudo -n podman info): is podman installed, and passwordless sudo allowed? ($1)"
    elif [[ "$CONTAINER_RT" == podman ]]; then
      die "cannot talk to podman (podman info): is podman installed for this user? ($1)"
    else
      die "cannot talk to the docker daemon (is your user in the docker group?) ($1)"
    fi
  }
}

# rt_save: a single image as a docker-archive on stdout, for the worker's `ctr load` (podman's default archive format
# is not the one docker load reads; keep both ends interchangeable while a pair moves from one runtime to the other).
rt_save() {
  if [[ "$CONTAINER_RT" == podman ]]; then ctr save --format docker-archive "$1"; else ctr save "$1"; fi
}

# rt_build_net: the extra `build` arguments one runtime needs. Podman's build container has its own network: on a
# Spark the host resolves through systemd-resolved's 127.0.0.53 stub, which that network cannot reach, so the
# Dockerfile's first download would fail. Docker's build has the host's network already.
rt_build_net() { [[ "$CONTAINER_RT" == podman ]] && printf '%s' '--network host'; }

# remote_ctr: what a command sent to the worker needs for `ctr` to mean the same runtime there (nodes.sh's worker()
# puts it in front of every remote command, so a one-liner can say `ctr ps` and mean docker or podman).
remote_ctr() { printf '%s%s; ' "$(runtime_env)" "$(declare -f ctr)"; }

# rt_root: where the runtime keeps its images; the free-space checks count against it (docker: DockerRootDir,
# podman: Store.GraphRoot).
rt_root() {
  if [[ "${CONTAINER_RT:-docker}" == podman ]]; then
    ctr info -f '{{.Store.GraphRoot}}' 2>/dev/null || echo /var/lib/containers/storage
  else
    ctr info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker
  fi
}

# rt_has_gpu: whether the runtime is set up to hand a GPU to a container. Docker lists an nvidia runtime; podman goes
# through the NVIDIA CDI spec (prepare.sh checks nvidia.com/gpu=all separately).
rt_has_gpu() {
  if [[ "${CONTAINER_RT:-docker}" == podman ]]; then
    nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu'
  else
    ctr info 2>/dev/null | grep -qi nvidia
  fi
}

# rt_needs_root: whether this runtime's store is only readable by root (rootful podman: /var/lib/containers is 0700).
rt_needs_root() { [[ "${CONTAINER_RT:-docker}" == podman && "${CONTAINER_RT_SUDO:-0}" == 1 ]]; }

# rt_dev <path>: the filesystem's device id, read with the privilege the runtime runs as. A bare stat as the login
# user fails on rootful podman's store (EACCES on /var/lib/containers), and an empty value then poisons the
# arithmetic in prepare.sh's free-space checks.
rt_dev() {
  if rt_needs_root; then sudo -n stat -c %d "$1" 2>/dev/null; else stat -c %d "$1" 2>/dev/null; fi
}

# rt_free_gb <path>: whole GiB available, same privilege rule (docker's root is readable by the docker group).
rt_free_gb() {
  if rt_needs_root; then sudo -n df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
  else df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; fi
}
