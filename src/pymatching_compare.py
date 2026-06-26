from __future__ import annotations

import random

import pandas as pd

from src.decoder import BruteForceMWPMDecoder
from src.lattice import Edge, Node, PlanarSurfaceCodeLattice, xor_edges
from src.noise import sample_iid_edge_noise


def _require_pymatching():
    try:
        import numpy as np
        import pymatching
    except ImportError as exc:
        raise RuntimeError(
            "PyMatching comparison requires optional dependencies. "
            "Install them with: pip install -r requirements-pymatching.txt"
        ) from exc
    return np, pymatching


def build_pymatching_decoder(lattice: PlanarSurfaceCodeLattice):
    np, pymatching = _require_pymatching()
    del np

    nodes = lattice.nodes()
    node_to_id = {node: i for i, node in enumerate(nodes)}
    edges = lattice.data_edges()

    matching = pymatching.Matching()
    for fault_id, edge in enumerate(edges):
        a, b = lattice.edge_endpoints(edge)
        if a is not None and b is not None:
            matching.add_edge(node_to_id[a], node_to_id[b], fault_ids={fault_id}, weight=1.0)
        else:
            node = a if a is not None else b
            assert node is not None
            matching.add_boundary_edge(node_to_id[node], fault_ids={fault_id}, weight=1.0)

    return matching, nodes, edges


def decode_with_pymatching(lattice: PlanarSurfaceCodeLattice, syndrome: set[Node]) -> set[Edge]:
    np, _ = _require_pymatching()
    matching, nodes, edges = build_pymatching_decoder(lattice)
    syndrome_vector = np.array([node in syndrome for node in nodes], dtype=np.uint8)
    predicted = matching.decode(syndrome_vector)
    return {edge for edge, bit in zip(edges, predicted) if bit}


def compare_from_scratch_with_pymatching(
    distance: int,
    physical_error_rate: float,
    shots: int,
    seed: int = 7,
    error_type: str = "z",
) -> pd.DataFrame:
    _require_pymatching()
    rng = random.Random(seed)
    lattice = PlanarSurfaceCodeLattice(distance=distance, error_type=error_type)  # type: ignore[arg-type]
    from_scratch = BruteForceMWPMDecoder(lattice)

    records = []
    for shot_id in range(shots):
        sample = sample_iid_edge_noise(lattice, physical_error_rate, rng)

        fs_decoded = from_scratch.decode(sample.syndrome)
        pm_correction = decode_with_pymatching(lattice, sample.syndrome)

        fs_residual = xor_edges(sample.error_edges, fs_decoded.correction_edges)
        pm_residual = xor_edges(sample.error_edges, pm_correction)

        records.append(
            {
                "shot_id": shot_id,
                "distance": distance,
                "error_type": error_type,
                "physical_error_rate": physical_error_rate,
                "from_scratch_logical_failure": lattice.logical_parity(fs_residual),
                "pymatching_logical_failure": lattice.logical_parity(pm_residual),
                "from_scratch_correction_weight": len(fs_decoded.correction_edges),
                "pymatching_correction_weight": len(pm_correction),
                "same_correction": fs_decoded.correction_edges == pm_correction,
            }
        )

    return pd.DataFrame(records)
