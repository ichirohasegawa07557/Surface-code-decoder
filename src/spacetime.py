from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import pandas as pd

from src.lattice import Boundary, Edge, Node, PlanarSurfaceCodeLattice


class Node3D(NamedTuple):
    """A syndrome defect location in a 3D space-time decoding graph."""

    time: int
    row: int
    col: int


@dataclass(frozen=True)
class SpaceTimeMatch:
    a: Node3D
    b: Node3D | None
    boundary: str | None
    weight: int


@dataclass(frozen=True)
class SpaceTimeDecodeResult:
    matches: tuple[SpaceTimeMatch, ...]
    total_weight: int


@dataclass(frozen=True)
class SpaceTimeNoiseSample:
    data_error_edges: set[tuple[int, Edge]]
    measurement_error_edges: set[tuple[int, Node]]
    defects: set[Node3D]


class SpaceTimeDecodingGraph:
    """Phenomenological 3D space-time decoding graph.

    This graph adds a time coordinate to the 2D decoding lattice. Spatial data
    errors create defects inside a single time layer. Measurement errors create
    pairs of defects at the same spatial check in adjacent time layers.

    The implementation is educational and intentionally compact. It is meant to
    show how measurement rounds turn a 2D matching problem into a 3D matching
    problem.
    """

    def __init__(self, distance: int, rounds: int, error_type: str = "z"):
        if rounds < 2:
            raise ValueError("rounds must be at least 2 for a space-time graph")
        self.distance = distance
        self.rounds = rounds
        self.lattice = PlanarSurfaceCodeLattice(distance=distance, error_type=error_type)  # type: ignore[arg-type]

    def toggle(self, defects: set[Node3D], node: Node3D) -> None:
        if node in defects:
            defects.remove(node)
        else:
            defects.add(node)

    def sample_noise(
        self,
        data_error_rate: float,
        measurement_error_rate: float,
        rng: random.Random,
    ) -> SpaceTimeNoiseSample:
        if not 0 <= data_error_rate <= 1:
            raise ValueError("data_error_rate must be between 0 and 1")
        if not 0 <= measurement_error_rate <= 1:
            raise ValueError("measurement_error_rate must be between 0 and 1")

        defects: set[Node3D] = set()
        data_errors: set[tuple[int, Edge]] = set()
        measurement_errors: set[tuple[int, Node]] = set()

        # Spatial data errors inside each round.
        for t in range(self.rounds):
            for edge in self.lattice.data_edges():
                if rng.random() < data_error_rate:
                    data_errors.add((t, edge))
                    for endpoint in self.lattice.edge_endpoints(edge):
                        if endpoint is not None:
                            self.toggle(defects, Node3D(t, endpoint.row, endpoint.col))

        # Measurement errors connect the same check between adjacent rounds.
        for t in range(self.rounds - 1):
            for node in self.lattice.nodes():
                if rng.random() < measurement_error_rate:
                    measurement_errors.add((t, node))
                    self.toggle(defects, Node3D(t, node.row, node.col))
                    self.toggle(defects, Node3D(t + 1, node.row, node.col))

        return SpaceTimeNoiseSample(data_errors, measurement_errors, defects)

    def distance_between(self, a: Node3D, b: Node3D) -> int:
        return abs(a.time - b.time) + abs(a.row - b.row) + abs(a.col - b.col)

    def distance_to_boundary(self, node: Node3D, boundary: str) -> int:
        if boundary in self.lattice.boundaries:
            return self.lattice.boundary_distance(Node(node.row, node.col), boundary)  # type: ignore[arg-type]
        if boundary == "past":
            return node.time + 1
        if boundary == "future":
            return self.rounds - node.time
        raise ValueError(f"unknown boundary: {boundary}")

    @property
    def boundaries(self) -> tuple[str, ...]:
        return (*self.lattice.boundaries, "past", "future")


class SpaceTimeMWPMDecoder:
    """Small exact MWPM decoder for the 3D educational graph."""

    def __init__(self, graph: SpaceTimeDecodingGraph):
        self.graph = graph

    def decode(self, defects: set[Node3D]) -> SpaceTimeDecodeResult:
        ordered = tuple(sorted(defects))
        n = len(ordered)

        @lru_cache(maxsize=None)
        def solve(mask: int) -> tuple[int, tuple[SpaceTimeMatch, ...]]:
            if mask == 0:
                return 0, ()

            i = (mask & -mask).bit_length() - 1
            a = ordered[i]
            remaining = mask & ~(1 << i)

            best_cost = 10**9
            best_matches: tuple[SpaceTimeMatch, ...] = ()

            for boundary in self.graph.boundaries:
                w = self.graph.distance_to_boundary(a, boundary)
                sub_cost, sub_matches = solve(remaining)
                total = w + sub_cost
                if total < best_cost:
                    best_cost = total
                    best_matches = (SpaceTimeMatch(a=a, b=None, boundary=boundary, weight=w),) + sub_matches

            j_mask = remaining
            while j_mask:
                j = (j_mask & -j_mask).bit_length() - 1
                b = ordered[j]
                w = self.graph.distance_between(a, b)
                next_mask = remaining & ~(1 << j)
                sub_cost, sub_matches = solve(next_mask)
                total = w + sub_cost
                if total < best_cost:
                    best_cost = total
                    best_matches = (SpaceTimeMatch(a=a, b=b, boundary=None, weight=w),) + sub_matches
                j_mask &= j_mask - 1

            return best_cost, best_matches

        total, matches = solve((1 << n) - 1)
        return SpaceTimeDecodeResult(matches=matches, total_weight=total)


def defects_to_dataframe(defects: set[Node3D]) -> pd.DataFrame:
    return pd.DataFrame([node._asdict() for node in sorted(defects)])
