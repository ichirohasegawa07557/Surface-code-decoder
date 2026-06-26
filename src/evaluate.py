from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from tqdm import tqdm

from src.decoder import make_decoder
from src.lattice import ErrorType, PlanarSurfaceCodeLattice, xor_edges
from src.noise import sample_css_iid_noise, sample_iid_edge_noise

DecoderName = Literal["mwpm", "union_find"]


@dataclass(frozen=True)
class ShotResult:
    logical_failure: bool
    syndrome_weight: int
    matching_weight: int
    error_weight: int
    correction_weight: int


@dataclass(frozen=True)
class CSSShotResult:
    logical_failure: bool
    x_logical_failure: bool
    z_logical_failure: bool
    x_syndrome_weight: int
    z_syndrome_weight: int
    x_matching_weight: int
    z_matching_weight: int


def run_single_shot(
    distance: int,
    physical_error_rate: float,
    rng: random.Random,
    error_type: ErrorType = "z",
    decoder_name: str = "mwpm",
) -> ShotResult:
    lattice = PlanarSurfaceCodeLattice(distance=distance, error_type=error_type)
    decoder = make_decoder(decoder_name, lattice)

    sample = sample_iid_edge_noise(
        lattice=lattice,
        physical_error_rate=physical_error_rate,
        rng=rng,
    )

    decoded = decoder.decode(sample.syndrome)
    residual_chain = xor_edges(sample.error_edges, decoded.correction_edges)
    logical_failure = lattice.logical_parity(residual_chain) == 1

    return ShotResult(
        logical_failure=logical_failure,
        syndrome_weight=len(sample.syndrome),
        matching_weight=decoded.total_weight,
        error_weight=len(sample.error_edges),
        correction_weight=len(decoded.correction_edges),
    )


def run_css_single_shot(
    distance: int,
    physical_error_rate: float,
    rng: random.Random,
    decoder_name: str = "mwpm",
) -> CSSShotResult:
    css_sample = sample_css_iid_noise(
        distance=distance,
        x_error_rate=physical_error_rate,
        z_error_rate=physical_error_rate,
        rng=rng,
    )

    x_lattice = PlanarSurfaceCodeLattice(distance=distance, error_type="x")
    z_lattice = PlanarSurfaceCodeLattice(distance=distance, error_type="z")
    x_decoder = make_decoder(decoder_name, x_lattice)
    z_decoder = make_decoder(decoder_name, z_lattice)

    x_decoded = x_decoder.decode(css_sample.x_errors.syndrome)
    z_decoded = z_decoder.decode(css_sample.z_errors.syndrome)

    x_residual = xor_edges(css_sample.x_errors.error_edges, x_decoded.correction_edges)
    z_residual = xor_edges(css_sample.z_errors.error_edges, z_decoded.correction_edges)

    x_failure = x_lattice.logical_parity(x_residual) == 1
    z_failure = z_lattice.logical_parity(z_residual) == 1

    return CSSShotResult(
        logical_failure=x_failure or z_failure,
        x_logical_failure=x_failure,
        z_logical_failure=z_failure,
        x_syndrome_weight=len(css_sample.x_errors.syndrome),
        z_syndrome_weight=len(css_sample.z_errors.syndrome),
        x_matching_weight=x_decoded.total_weight,
        z_matching_weight=z_decoded.total_weight,
    )


def run_experiment(
    distances: list[int],
    physical_error_rates: list[float],
    shots: int,
    seed: int = 7,
    error_type: ErrorType = "z",
    decoder_name: str = "mwpm",
) -> pd.DataFrame:
    """Estimate logical failure rates for one CSS component."""

    rng = random.Random(seed)
    records = []

    for distance in distances:
        for p in physical_error_rates:
            failures = 0
            syndrome_weights = []
            matching_weights = []
            error_weights = []
            correction_weights = []

            iterator = tqdm(
                range(shots),
                desc=f"{decoder_name}, {error_type}, d={distance}, p={p:.3f}",
                leave=False,
            )

            for _ in iterator:
                shot = run_single_shot(
                    distance=distance,
                    physical_error_rate=p,
                    rng=rng,
                    error_type=error_type,
                    decoder_name=decoder_name,
                )
                failures += int(shot.logical_failure)
                syndrome_weights.append(shot.syndrome_weight)
                matching_weights.append(shot.matching_weight)
                error_weights.append(shot.error_weight)
                correction_weights.append(shot.correction_weight)

            records.append(
                {
                    "decoder": decoder_name,
                    "error_type": error_type,
                    "distance": distance,
                    "physical_error_rate": p,
                    "shots": shots,
                    "logical_failures": failures,
                    "logical_failure_rate": failures / shots,
                    "avg_syndrome_weight": sum(syndrome_weights) / shots,
                    "avg_matching_weight": sum(matching_weights) / shots,
                    "avg_error_weight": sum(error_weights) / shots,
                    "avg_correction_weight": sum(correction_weights) / shots,
                }
            )

    return pd.DataFrame(records)


def run_css_experiment(
    distances: list[int],
    physical_error_rates: list[float],
    shots: int,
    seed: int = 7,
    decoder_name: str = "mwpm",
) -> pd.DataFrame:
    """Estimate logical failure rates for CSS-style X/Z separated decoding."""

    rng = random.Random(seed)
    records = []

    for distance in distances:
        for p in physical_error_rates:
            failures = 0
            x_failures = 0
            z_failures = 0
            x_syndrome_weights = []
            z_syndrome_weights = []
            x_matching_weights = []
            z_matching_weights = []

            iterator = tqdm(
                range(shots),
                desc=f"CSS {decoder_name}, d={distance}, p={p:.3f}",
                leave=False,
            )

            for _ in iterator:
                shot = run_css_single_shot(
                    distance=distance,
                    physical_error_rate=p,
                    rng=rng,
                    decoder_name=decoder_name,
                )
                failures += int(shot.logical_failure)
                x_failures += int(shot.x_logical_failure)
                z_failures += int(shot.z_logical_failure)
                x_syndrome_weights.append(shot.x_syndrome_weight)
                z_syndrome_weights.append(shot.z_syndrome_weight)
                x_matching_weights.append(shot.x_matching_weight)
                z_matching_weights.append(shot.z_matching_weight)

            records.append(
                {
                    "decoder": decoder_name,
                    "mode": "css_xz_separated",
                    "distance": distance,
                    "physical_error_rate": p,
                    "shots": shots,
                    "logical_failures": failures,
                    "logical_failure_rate": failures / shots,
                    "x_logical_failures": x_failures,
                    "x_logical_failure_rate": x_failures / shots,
                    "z_logical_failures": z_failures,
                    "z_logical_failure_rate": z_failures / shots,
                    "avg_x_syndrome_weight": sum(x_syndrome_weights) / shots,
                    "avg_z_syndrome_weight": sum(z_syndrome_weights) / shots,
                    "avg_x_matching_weight": sum(x_matching_weights) / shots,
                    "avg_z_matching_weight": sum(z_matching_weights) / shots,
                }
            )

    return pd.DataFrame(records)
