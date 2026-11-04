# Podman instead of Docker

The recipe defaults to Docker and is written for it, but every container command goes through one abstraction, so the
same scripts run the pair under Podman. Three settings do it:

```bash
CONTAINER_RT=podman        # docker (default) or podman
CONTAINER_RT_SUDO=1        # rootful podman ("sudo -n podman")
CGROUP_PARENT=tensorfold.slice   # optional: the system slice each rank joins
```

Put them in `scripts/local.sh` (or `.env`, with `local.sh` winning). `CONTAINER_RT=docker` needs no other change and
is what everyone who does not set these runs.

## Why rootful

Rootless Podman cannot do two things this recipe needs on a Spark pair:

- **the worker's read-only NFS volume**: the local volume driver mounts NFS, and only root may mount,
- **a system cgroup slice**: `CGROUP_PARENT` (`tensorfold.slice`, with `MemorySwapMax=0` from the host tuning) exists
  under root's cgroup tree. A ~110 GiB rank on a 121 GiB node that is allowed to swap is a node that stops answering.

`CONTAINER_RT_SUDO=1` makes every call `sudo -n podman`; passwordless sudo for the login user is required on both
Sparks. With `CONTAINER_RT=podman` and `CONTAINER_RT_SUDO=0` the image-only helpers work, but `prepare.sh` will fail
on the NFS volume when `WORKER_WEIGHTS=nfs`.

## Moving a pair that already runs on Docker

The image is content-addressed by `image_hash()` (the patches plus `IMAGE_EXTRAS`), not by the container runtime, so a
Podman pair does not rebuild or re-download anything:

```bash
# head
sudo -n docker save tensorfold-glm53:<tag> | sudo -n podman load

# worker (the podman archive format is the one docker load reads too, should the pair ever go back)
sudo -n podman save --format docker-archive tensorfold-glm53:<tag> | ssh <worker> 'sudo -n podman load'
```

`scripts/nodes.sh` identifies an image by `RootFS.Layers` + `Config` (`image_ident`), which both runtimes report
identically, so `prepare.sh` sees the image as present on both ends and skips the copy. The kernel cache, the prepared
marker and the checkpoint are runtime-independent; the NFS volume is not (separate runtime, separate volume store),
so let `ensure_nfs_volume` create it again under Podman.

The GHCR image can simply be pulled instead: `./start.sh` with `PULL=1` (the default) pulls the pinned digest with
`podman pull`.

## The four differences (and the fifth)

1. **Build network.** `podman build --network host`. The build container has its own network and cannot reach
   systemd-resolved's `127.0.0.53` stub, so the Dockerfile's first download fails to resolve. `rt_build_net` adds the
   flag; Docker's build already uses the host's network.
2. **Archive format.** `podman save --format docker-archive` when the other end may still be Docker (`rt_save`).
3. **`--shm-size` with `--ipc=host`.** Docker accepts the pair; Podman refuses it ("cannot set shm size when IPC
   namespace is host"). `start.sh` adds `--shm-size 16g` only for Docker. Nothing is lost: host IPC already gives the
   container the host's `/dev/shm` (61 GiB on a Spark).
4. **The NFS volume's device.** Docker spells the export `device=:/path` with `o=addr=<server>,…`; Podman hands
   `device` to `mount.nfs`, so it must be `device=<server>:/path`. Given Docker's form, Podman creates the volume and
   then fails at container start with `mount.nfs: Failed to resolve server :`. `nfs_volume_args` /
   `nfs_volume_matches` hold both spellings; a volume created by the other runtime is reported with the command that
   removes it.
5. **The image store's root.** `docker info -f '{{.DockerRootDir}}'` has no Podman equivalent; the free-space checks
   (`MIN_FREE_GB`, `IMAGE_FREE_GB`) measure `podman info -f '{{.Store.GraphRoot}}'` (`rt_root`).

## What stays the same

The rank command line, `--gpus all` (Podman maps it through the NVIDIA CDI spec), `--ipc=host`, `--network host`,
`/dev/infiniband`, `CAP_IPC_LOCK` and the memlock/stack ulimits, the container name on both Sparks, `logs`, `inspect`,
`ps --format`, `top`, `volume inspect/create`, `image inspect`, `tag`, `pull`, `build` and `login/push`
(`--authfile` instead of `DOCKER_CONFIG`). `stop.sh` still saves each rank's log through `podman logs` before it
removes the container.

## Verifying

```bash
./start.sh                                   # prints the runtime it uses and every command it runs
sudo -n podman ps --format '{{.Names}} {{.Status}}'         # both Sparks: glm53-flash-tf
sudo -n podman inspect -f '{{.HostConfig.CgroupParent}}' glm53-flash-tf
sudo -n docker ps                            # empty: nothing left behind in the docker store
curl -s http://127.0.0.1:8000/v1/models
```

`DRY_RUN=1 ./start.sh` prints every command without running any of them, and works before anything is installed.
