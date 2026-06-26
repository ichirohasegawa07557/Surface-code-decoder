from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.lattice import Boundary, Edge, Node, PlanarSurfaceCodeLattice


@dataclass(frozen=True)
class Match:
    """One matching decision made by a decoder."""

    a: Node
    b: Node | None
    boundary: Boundary | None
    weight: int


@dataclass(frozen=True)
class DecodeResult:
    """Decoded correction chain and matching metadata."""

    correction_edges: set[Edge]
    matches: tuple[Match, ...]
    total_weight: int


class BruteForceMWPMDecoder:
    """Minimum-weight decoder implemented from scratch.

    The decoder solves a small matching problem by dynamic programming over a
    bit mask of syndrome defects. It is exact for the simplified decoding graph
    used in this repository, but it is intended for small educational examples,
    not for large-scale QEC simulations.
    """

    def __init__(self, lattice: PlanarSurfaceCodeLattice):
        self.lattice = lattice

    def decode(self, syndrome: set[Node]) -> DecodeResult:
        defects = tuple(sorted(syndrome))
        n = len(defects)

        @lru_cache(maxsize=None)
        def solve(mask: int) -> tuple[int, tuple[Match, ...]]:
            if mask == 0:
                return 0, ()

            # Pick the first remaining defect.
            i = (mask & -mask).bit_length() - 1
            a = defects[i]
            remaining_without_i = mask & ~(1 << i)

            best_cost = 10**9
            best_matches: tuple[Match, ...] = ()

            # Option 1: pair the defect with an allowed boundary.
            for boundary in self.lattice.boundaries:
                w = self.lattice.boundary_distance(a, boundary)
                sub_cost, sub_matches = solve(remaining_without_i)
                total = w + sub_cost
                if total < best_cost:
                    best_cost = total
                    best_matches = (Match(a=a, b=None, boundary=boundary, weight=w),) + sub_matches

            # Option 2: pair the defect with another defect.
            j_mask = remaining_without_i
            while j_mask:
                j = (j_mask & -j_mask).bit_length() - 1
                b = defects[j]
                w = self.lattice.distance_between(a, b)
                next_mask = remaining_without_i & ~(1 << j)
                sub_cost, sub_matches = solve(next_mask)
                total = w + sub_cost
                if total < best_cost:
                    best_cost = total
                    best_matches = (Match(a=a, b=b, boundary=None, weight=w),) + sub_matches
                j_mask &= j_mask - 1

            return best_cost, best_matches

        full_mask = (1 << n) - 1
        total_weight, matches = solve(full_mask)
        correction_edges = matches_to_edges(self.lattice, matches)

        return DecodeResult(
            correction_edges=correction_edges,
            matches=matches,
            total_weight=total_weight,
        )


class DisjointSet:
    """Small union-find data structure used by UnionFindDecoder."""

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class UnionFindDecoder:
    """Educational Union-Find style decoder.

    This is not a production implementation of the Delfosse-Nickerson decoder.
    It is a compact from-scratch decoder that uses a disjoint-set structure to
    build neutral clusters by connecting odd clusters through short paths or to
    an allowed boundary. It is useful for comparing the behavior of an exact
    brute-force MWPM decoder with a fast approximate clustering decoder.
    """

    def __init__(self, lattice: PlanarSurfaceCodeLattice):
        self.lattice = lattice

    def decode(self, syndrome: set[Node]) -> DecodeResult:
        defects = tuple(sorted(syndrome))
        n = len(defects)
        if n == 0:
            return DecodeResult(correction_edges=set(), matches=(), total_weight=0)

        boundary_ids = {boundary: n + i for i, boundary in enumerate(self.lattice.boundaries)}
        total_items = n + len(boundary_ids)
        dsu = DisjointSet(total_items)

        # Cluster metadata keyed by current root.
        parity = {i: 1 for i in range(n)}
        has_boundary = {i: False for i in range(n)}
        for boundary, idx in boundary_ids.items():
            parity[idx] = 0
            has_boundary[idx] = True

        candidates: list[tuple[int, int, int, Match]] = []

        # Defect-defect candidate connections.
        for i, a in enumerate(defects):
            for j in range(i + 1, n):
                b = defects[j]
                w = self.lattice.distance_between(a, b)
                candidates.append((w, i, j, Match(a=a, b=b, boundary=None, weight=w)))

        # Defect-boundary candidate connections.
        for i, a in enumerate(defects):
            for boundary, b_id in boundary_ids.items():
                w = self.lattice.boundary_distance(a, boundary)
                candidates.append((w, i, b_id, Match(a=a, b=None, boundary=boundary, weight=w)))

        candidates.sort(key=lambda item: item[0])
        chosen: list[Match] = []

        def root_state(root: int) -> tuple[int, bool]:
            root = dsu.find(root)
            return parity.get(root, 0), has_boundary.get(root, False)

        def unresolved_roots() -> set[int]:
            roots = {dsu.find(i) for i in range(total_items)}
            return {r for r in roots if parity.get(r, 0) == 1 and not has_boundary.get(r, False)}

        for _, u, v, match in candidates:
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                continue

            pu, bu = root_state(ru)
            pv, bv = root_state(rv)

            # Skip connections between already-neutral clusters unless one side
            # still needs a boundary to neutralize an odd cluster.
            if (pu == 0 or bu) and (pv == 0 or bv):
                continue

            new_root = dsu.union(ru, rv)
            old_roots = {ru, rv}
            new_parity = pu ^ pv
            new_boundary = bu or bv
            for r in old_roots:
                parity.pop(r, None)
                has_boundary.pop(r, None)
            parity[new_root] = new_parity
            has_boundary[new_root] = new_boundary
            chosen.append(match)

            if not unresolved_roots():
                break

        correction_edges = matches_to_edges(self.lattice, tuple(chosen))
        total_weight = sum(match.weight for match in chosen)
        return DecodeResult(correction_edges=correction_edges, matches=tuple(chosen), total_weight=total_weight)


def make_decoder(name: str, lattice: PlanarSurfaceCodeLattice):
    name = name.lower().replace("-", "_")
    if name in {"mwpm", "bruteforce", "brute_force"}:
        return BruteForceMWPMDecoder(lattice)
    if name in {"union_find", "unionfind", "uf"}:
        return UnionFindDecoder(lattice)
    raise ValueError("decoder name must be 'mwpm' or 'union_find'")


def matches_to_edges(lattice: PlanarSurfaceCodeLattice, matches: tuple[Match, ...]) -> set[Edge]:
    correction_edges: set[Edge] = set()

    for match in matches:
        if match.b is not None:
            path = lattice.path_between(match.a, match.b)
        else:
            assert match.boundary is not None
            path = lattice.path_to_boundary(match.a, match.boundary)

        # Add modulo 2.
        for edge in path:
            if edge in correction_edges:
                correction_edges.remove(edge)
            else:
                correction_edges.add(edge)

    return correction_edges
