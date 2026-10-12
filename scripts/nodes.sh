# The Sparks: this machine (rank 0), and the workers over ssh: WORKER (rank 1), WORKER2 (rank 2).
# Each node's links to the others are found from the routes and subnets between them.
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.

# wvar <NAME> <i>: the name of worker i's own setting: NAME for worker 1 (WORKER, FABRIC_PEER, NFS_SERVER,
# WORKER_WEIGHTS, WORKER_HF_CACHE), NAME<i> for the others (WORKER2, FABRIC_PEER2, NFS_SERVER2, WORKER_WEIGHTS2,
# WORKER_HF_CACHE2).
wvar() { if (( $2 == 1 )); then echo "$1"; else echo "$1$2"; fi; }
wval() { local v; v=$(wvar "$1" "$2"); echo "${!v:-}"; }
worker_host() { wval WORKER "$1"; }
# The workers this start uses (1 .. TP-1), and every worker configured at all (stop.sh stops them all).
worker_ids() { seq 1 $((TP - 1)); }
configured_workers() { local i; for i in 1 2; do [[ -z "$(worker_host "$i")" ]] || echo "$i"; done; }
# worker i's weights: WORKER_WEIGHTS for worker 1, WORKER_WEIGHTS<i> (default: WORKER_WEIGHTS) for the others
worker_weights() { local w; w=$(wval WORKER_WEIGHTS "$1"); echo "${w:-$WORKER_WEIGHTS}"; }

# check_workers: TP is 2 or 3, and WORKER .. WORKER<TP-1> are set and distinct (later ones are left out).
check_workers() {
  local i j h
  [[ "$TP" =~ ^[23]$ ]] || die "TP is 2 (./start.sh) or 3 (./start-tp3.sh, experimental) Sparks, not $TP"
  for i in $(worker_ids); do
    h=$(worker_host "$i")
    [[ -n "$h" ]] || die "TP=$TP needs $((TP - 1)) workers: set $(wvar WORKER "$i")=user@<address of rank $i> in scripts/local.sh (see scripts/local.sh.example)"
    [[ "$(worker_weights "$i")" =~ ^(copy|nfs)$ ]] || die "$(wvar WORKER_WEIGHTS "$i") is copy or nfs, not $(worker_weights "$i")"
    for (( j = 1; j < i; j++ )); do
      [[ "$h" != "$(worker_host "$j")" ]] || die "$(wvar WORKER "$i") and $(wvar WORKER "$j") are the same node ($h)"
    done
  done
}

# DRY_RUN=1 goes on without workers it cannot reach (a fictitious WORKER2): their values print as <WORKER2:...>.
declare -A WORKER_DOWN=()
# worker <i> <command...>: run a command on worker i (rank i). Key-based ssh only, no prompts.
worker() {
  local i=$1; shift
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$(worker_host "$i")" "$@"
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log <i>: the same, for rank i on worker i (into ~/.cache/tensorfold-glm53/logs there)
worker_save_log() {
  worker "$1" "$(declare -f save_log); save_log \"\$HOME/.cache/tensorfold-glm53/logs\" $1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  local i=$1 v h
  v=$(wvar WORKER "$i"); h=$(worker_host "$i")
  [[ -n "$h" ]] || die "$v is not set: put $v=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker "$i" true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $h ($v): its values print as <$v:...>"
    WORKER_DOWN[$i]=1
    return 0
  fi
  die "cannot ssh to $h without a password: set up key-based ssh (ssh-copy-id $h)"
}
need_workers() { local i; for i in $(worker_ids); do need_worker "$i"; done; }

# Worker i's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank i. WORKER_HF_CACHE (worker 1) / WORKER_HF_CACHE<i> wins: a worker whose
# cache is not its HF_HOME (e.g. a shared models disk).
worker_hf_cache() {
  local c; c=$(wval WORKER_HF_CACHE "$1")
  if [[ -n "$c" ]]; then echo "$c"; return; fi
  [[ -z "${WORKER_DOWN[$1]:-}" ]] || { echo "<$(wvar WORKER "$1"):hf-cache>"; return 0; }
  worker "$1" 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# Weights over NFS: rank i mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on worker i, from
# nfs_server i: NFS_SERVER (worker 1) / NFS_SERVER<i>, else the head's address on that worker's link (detect_links).
# ensure_nfs_volume i creates it (or checks the one there names the same export); worker_nfs i <cmd...> runs a command
# in a throwaway container on worker i with the volume at /hf.
nfs_server() { local s; s=$(wval NFS_SERVER "$1"); echo "${s:-${LINK_HEAD_ADDR[$1]:-}}"; }
ensure_nfs_volume() {
  local i=$1 server have h
  server=$(nfs_server "$i"); h=$(worker_host "$i")
  [[ -n "$server" ]] || die "no NFS server address for $h: set $(wvar NFS_SERVER "$i") to the head's address on that worker's link"
  have=$(worker "$i" "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "$i" "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $h"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$h's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / $(wvar NFS_SERVER "$i")"
  fi
}
worker_nfs() { local i=$1; shift; worker "$i" "docker run --rm --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and NCCL gets one x4: ~112 Gb/s of the port's 200.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d $n ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: one rail's traffic reaches the peer's twin - a full-size, unfragmentable
# ping bound to that rail's own port (-I <netdev>: an address alone would be routed out of whichever port of the same
# subnet the kernel picks, usually the cabled one), so an unaddressed, differently-subnetted or smaller-MTU twin fails
# here instead of as vendor_err 0x81 (transport retry exceeded) in the first all-gather (issue #66). Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# The same function on worker i (its definition is sent over ssh).
worker_link_info() { worker "$1" "$(declare -f link_info); link_info $2"; }

# Per rank r, filled by detect_links: NODE_DEV[r] (the netdev of NCCL's bootstrap socket), NODE_HCAS[r] (its RoCE
# devices toward its peers), NODE_GID[r] (their RoCE v2 GID index; empty at TP>2 when they differ: NCCL then picks
# each device's own); LINK_HEAD_ADDR[i] / LINK_WORKER_ADDR[i]: the head's and worker i's addresses on their link (the
# head's is worker i's NFS server).
declare -a NODE_DEV=() NODE_HCAS=() NODE_GID=() LINK_HEAD_ADDR=() LINK_WORKER_ADDR=()

# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing". worker_image_ident <i> <image>: on worker i.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}

# cx7_peer <i>: worker i's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker "$1" 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# probe_rails: before a second rail (the PCIe twin of the cabled port, or a second cabled port) is used, check that it
# reaches the peer. roce.py pairs the two nodes' devices by list position at two ranks, so a twin that is up and has a GID
# but no path to the peer's twin (no address, another subnet, another MTU, a cable on a different port) ends in vendor_err
# 0x81. Rail k of the head is probed against rail k of the worker, in both directions; a rail that fails is dropped from both
# lists with a warning (the first device, the link itself, is never probed: the launcher reached the worker over it).
probe_rails() {
  local -a h w keep_h=() keep_w=()
  local k hdev hip hmtu wdev wip wmtu mtu=0 ok
  IFS=, read -ra h <<<"$HEAD_HCAS"; IFS=, read -ra w <<<"$WORKER_HCAS"
  (( ${#h[@]} > 1 && ${#h[@]} == ${#w[@]} )) || return 0
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { warn "DRY_RUN: the second rail ($HEAD_HCAS / $WORKER_HCAS) is not probed: the worker cannot be reached"; return 0; }
  keep_h=("${h[0]}"); keep_w=("${w[0]}")
  for (( k = 1; k < ${#h[@]}; k++ )); do
    read -r hdev hip hmtu <<<"$(rail_info "${h[k]}")"
    read -r wdev wip wmtu <<<"$(worker 1 "$(declare -f rail_info); rail_info ${w[k]}")"
    ok=0
    if [[ "$hip" =~ ^[0-9.]+$ && "$wip" =~ ^[0-9.]+$ && "$hmtu" =~ ^[0-9]+$ && "$wmtu" =~ ^[0-9]+$ ]]; then
      mtu=$(( hmtu < wmtu ? hmtu : wmtu ))
      if rail_ping "$hdev" "$wip" "$mtu" && worker 1 "$(declare -f rail_ping); rail_ping $wdev $hip $mtu"; then ok=1; fi
    fi
    if (( ok )); then
      keep_h+=("${h[k]}"); keep_w+=("${w[k]}")
      log "Rail ${h[k]} <-> ${w[k]}: $hip <-> $wip reaches the peer (MTU $mtu): used"
    else
      warn "second rail ${h[k]} (${hip:--}, MTU ${hmtu:--}) <-> ${w[k]} (${wip:--}, MTU ${wmtu:--}) does not reach the peer: NOT used, running on fewer rails. Give both twins an IPv4 address on one subnet per rail with the same MTU on both Sparks (README), or set NCCL_RAILS=1 to silence this"
    fi
  done
  HEAD_HCAS=$(IFS=,; echo "${keep_h[*]}"); WORKER_HCAS=$(IFS=,; echo "${keep_w[*]}")
}

# TP=2: HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the
# two ranks talk over (NCCL and the rendezvous). FABRIC_PEER overrides the worker's link address when WORKER is reached
# over another network.
detect_link() {
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer 1); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "${NCCL_RAILS:-2}" == 1 ]] || probe_rails
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# node_inventory [master]: this node's default-route netdev ("default <netdev>"), whether it routes to <master>
# ("master ok|none"), and one line per RoCE netdev that is up with an IPv4 address: "<netdev> <rdma device>
# <address/prefix> <RoCE v2 GID index of that address, or ->". Read-only (ip and sysfs).
node_inventory() {
  local n dev hca cidr want idx f
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in /sys/class/net/*; do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}

# pair_links <ranks...> (stdin: node_inventory lines, each prefixed with its rank): two nodes are linked where they
# have addresses in one subnet (a direct cable, or a switch). Prints "rank <r> <default netdev> <hcas> <gid or ->" (a
# node's RoCE devices toward all its peers, in its sysfs order; the GID index when it is one for all of them), "peer <i>
# <worker i's lowest address on its link to the head>" and "missing <a> <b>" for two ranks without a common subnet.
pair_links() {
  python3 -c '
import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) == 5:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4]))
ranks = [int(x) for x in sys.argv[1:]]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        used.update(id(e) for e in mine)
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
    hcas, gids = [], set()
    for e in nodes.get(r, []):              # in the node sysfs order
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
' "$@"
}

# TP>2: every node's RoCE netdevs, paired by subnet (pair_links); a node's NCCL devices are the union of its devices
# toward all its peers. NCCL's bootstrap socket goes over each node's default-route netdev (SOCKET_IFNAME overrides
# it): on a triangle every pair has its own subnet, so no CX7 netdev reaches both peers, and the 10.0.0.x-style
# addresses some setups route over the cables sit on lo. The rendezvous is MASTER_ADDR (config.sh).
detect_links() {
  if (( TP == 2 )); then detect_link; return; fi
  local r i peer kind a b c d pairs line missing="" names
  [[ -n "$MASTER_ADDR" ]] || die "no MASTER_ADDR: set it to an address of this node that every worker reaches (its LAN address)"
  local -a inv=()
  inv[0]=$(node_inventory)
  for i in $(worker_ids); do
    inv[i]=$(worker "$i" "$(declare -f node_inventory); node_inventory $MASTER_ADDR" 2>/dev/null || true)
    [[ "${inv[i]}" != *"master none"* ]] ||
      die "$(worker_host "$i") has no route to MASTER_ADDR $MASTER_ADDR: set MASTER_ADDR to an address of this node it reaches"
  done
  pairs=$(for r in "${!inv[@]}"; do sed "s/^/$r /" <<<"${inv[r]}"; done | pair_links 0 $(worker_ids)) ||
    die "could not pair the nodes' links"
  while read -r kind a b c d; do
    case "$kind" in
      rank)
        NODE_DEV[a]=${SOCKET_IFNAME:-$b}; NODE_HCAS[a]=$c; NODE_GID[a]=$d
        [[ "$d" != - ]] || NODE_GID[a]="" ;;
      peer)
        # the head's side of that link from its route to the worker's address there (FABRIC_PEER<i> overrides it)
        peer=$(wval FABRIC_PEER "$a"); peer=${peer:-$b}
        # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first, as at TP=2
        if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
          c=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
          [[ -n "$c" ]] || die "cannot resolve $peer ($(wvar FABRIC_PEER "$a")) to an IPv4 address"
          peer=$c
        fi
        read -r LINK_HEAD_ADDR[a] _ <<<"$(link_info "$peer")" || true
        LINK_WORKER_ADDR[a]=$peer ;;
      missing) missing+=" $a-$b" ;;
    esac
  done <<<"$pairs"
  for r in 0 $(worker_ids); do
    if [[ -n "${WORKER_DOWN[$r]:-}" ]]; then
      names=$(wvar WORKER "$r")
      NODE_DEV[r]="<$names:netdev>"; NODE_HCAS[r]="<$names:hcas>"; NODE_GID[r]="<$names:gid>"
      LINK_HEAD_ADDR[r]="<head address toward $names>"; LINK_WORKER_ADDR[r]="<$names:address>"
      continue
    fi
    names="this node"; (( r == 0 )) || names=$(worker_host "$r")
    [[ -n "${NODE_DEV[r]:-}" && "${NODE_DEV[r]}" != - ]] ||
      die "rank $r ($names) has no default route: set SOCKET_IFNAME to the netdev of the network all the Sparks share"
    [[ -n "${NODE_HCAS[r]:-}" && "${NODE_HCAS[r]}" != - ]] ||
      die "rank $r ($names) shares no RoCE subnet with any other rank: check the CX7 cabling and addresses (ip -4 addr)"
  done
  for line in $missing; do
    a=${line%-*}; b=${line#*-}
    if [[ -n "${WORKER_DOWN[$a]:-}${WORKER_DOWN[$b]:-}" ]]; then
      warn "DRY_RUN: the link between ranks $a and $b is unknown (a worker that cannot be reached)"
      continue
    fi
    die "ranks $a and $b share no RoCE subnet: $TP Sparks need a link between every pair (a triangle of direct cables; see README)"
  done
  return 0
}

# The NCCL settings of rank r's container (docker -e arguments).
# TP=2: the RoCE link's netdev, RoCE devices (both rails) and GID index, 4 channels. The RoCE all-gathers (patch 0006)
# take the same devices from NCCL_IB_HCA, as TF_ROCE_HCA is not set here (the command is v1.4's); TP>2 sets TF_ROCE_HCA
# per node (nccl_env_n).
# Measured between the Sparks: a decode-sized all-gather 32 us (80 with NCCL's defaults) and a prompt chunk's 32 MB
# 1.9 ms on two rails (3.5 on one). NCCL's other knobs (protocols, buffer sizes, QPs, NCCL_NET=IB and the like) were no
# better, and some of them kept NCCL on one rail.
nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  echo "-e NCCL_SOCKET_IFNAME=$dev -e NCCL_IB_HCA=$hcas -e NCCL_IB_GID_INDEX=$gid" \
       "-e NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-4} -e NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-4}" \
       ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG}
}
# TP>2, from the owner's 3-Spark vLLM recipe on this triangle: the bootstrap socket on the network every node shares;
# every RoCE device toward every peer; NCCL_CROSS_NIC=1 and subnet-aware routing (the image's NCCL 2.30.7 has it): each
# cable joins port 0 of one node to port 1 of the next, and NCCL otherwise pairs device index with device index
# (ibv_modify_qp ... Connection timed out); no P2P or SHM transport (one GPU a node). The GID index only when it is the
# same on all of a node's devices; else NCCL picks each device's RoCE v2 IPv4 entry itself.
nccl_env_n() {
  local dev=$1 hcas=$2 gid=$3
  echo "-e NCCL_SOCKET_IFNAME=$dev -e NCCL_IB_HCA=$hcas ${gid:+-e NCCL_IB_GID_INDEX=$gid}" \
       "-e NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1} -e NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1}" \
       "-e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1" \
       "-e NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-4} -e NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-4}" \
       "-e TF_ROCE_HCA=$hcas" \
       ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG}
}
rank_nccl_env() {
  if (( TP == 2 )); then nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}"
  else nccl_env_n "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}"; fi
}
