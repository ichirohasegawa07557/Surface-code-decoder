from __future__ import annotations

import random
from dataclasses import dataclass

from src.lattice import Edge, Node, PlanarSurfaceCodeLattice


@dataclass(frozen=True)
class NoiseSample:
    """One sampled error configuration and its syndrome."""

    error_edges: set[Edge]
    syndrome: set[Node]


@dataclass(frozen=True)
class CSSNoiseSample:
    """Independent X and Z error samples for CSS-style decoding."""

    x_errors: NoiseSample
    z_errors: NoiseSample


def sample_iid_edge_noise(
    lattice: PlanarSurfaceCodeLattice,
    physical_error_rate: float,
    rng: random.Random,
) -> NoiseSample:
    """Sample independent elementary errors on one decoding graph."""

    if not 0.0 <= physical_error_rate <= 1.0:
        raise ValueError("physical_error_rate must be between 0 and 1")

    error_edges: set[Edge] = set()
    for edge in lattice.data_edges():
        if rng.random() < physical_error_rate:
            error_edges.add(edge)

    syndrome = lattice.toggle_syndrome_from_edges(error_edges)
    return NoiseSample(error_edges=error_edges, syndrome=syndrome)


def sample_css_iid_noise(
    distance: int,
    x_error_rate: float,
    z_error_rate: float,
    rng: random.Random,
) -> CSSNoiseSample:
    """Sample independent X and Z errors for the two CSS components."""

    x_lattice = PlanarSurfaceCodeLattice(distance=distance, error_type="x")
    z_lattice = PlanarSurfaceCodeLattice(distance=distance, error_type="z")

    return CSSNoiseSample(
        x_errors=sample_iid_edge_noise(x_lattice, x_error_rate, rng),
        z_errors=sample_iid_edge_noise(z_lattice, z_error_rate, rng),
    )
