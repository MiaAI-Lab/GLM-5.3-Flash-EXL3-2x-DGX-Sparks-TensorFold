"""RoCE routes on four Sparks cabled as a ring (TF_ROCE_RING, patch 0099): the ranks across the ring need no path,
every neighbour does, and both PCIe twins of a cable pair up."""
import ipaddress

import pytest

from tensorfold.cuda.roce import plan_routes


def ip(a: str) -> int:
    return int(ipaddress.IPv4Address(a))


# the tested ring: spark N's rocep1s0f0, rocep1s0f1, roceP2p1s0f0, roceP2p1s0f1 (port 0, port 1, their twins), each
# cable's /24 with .N on the port and .10N on its twin
CABLES = {(0, 1): "10.0.22", (1, 2): "10.0.33", (2, 3): "10.0.34", (3, 0): "10.0.41"}
PORT = {(0, 1): (1, 1), (1, 2): (0, 0), (2, 3): (1, 1), (3, 0): (0, 0)}   # (port of a, port of b) a cable


def ring_addrs() -> list[list[tuple[int, int] | None]]:
    addrs: list[list[tuple[int, int] | None]] = [[None] * 4 for _ in range(4)]
    for (a, b), net in CABLES.items():
        pa, pb = PORT[(a, b)]
        for r, port in ((a, pa), (b, pb)):
            addrs[r][port] = (ip(f"{net}.{r + 1}"), 24)
            addrs[r][2 + port] = (ip(f"{net}.{r + 101}"), 24)
    return addrs


def test_ring_routes_neighbours_only():
    routes, stripes = plan_routes(ring_addrs(), ring=True)
    assert stripes == 2
    for r in range(4):
        for d in (1, 3):
            p = (r + d) % 4
            assert len(routes[(r, p)]) == 2
            # mirrored on the two ends
            assert sorted((b, a) for a, b in routes[(p, r)]) == sorted(routes[(r, p)])
        assert (r, (r + 2) % 4) not in routes


def test_ring_without_ring_mode_refused():
    with pytest.raises(RuntimeError, match="TF_ROCE_RING"):
        plan_routes(ring_addrs())


def test_ring_mode_still_needs_every_neighbour():
    addrs = ring_addrs()
    addrs[1] = [None if a is not None and (a[0] >> 8) == ip("10.0.22.0") >> 8 else a for a in addrs[1]]
    with pytest.raises(RuntimeError, match="0-1"):
        plan_routes(addrs, ring=True)


def test_ring_mode_needs_four_ranks():
    addrs = [[(ip("10.0.9.1"), 24)], [(ip("10.0.9.2"), 24)], [(ip("10.0.9.3"), 24)]]
    with pytest.raises(RuntimeError, match="four"):
        plan_routes(addrs, ring=True)
