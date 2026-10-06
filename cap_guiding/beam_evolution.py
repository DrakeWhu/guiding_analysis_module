"""Physical beam reductions and temporal aggregation, independent of plotting.

Positions: m in particle arrays, um in products. u = p/(m_e*c).
Frame axis: absolute guiding mesh z_max_um (last mesh sample, not a crossing).
This module never infers a loss mechanism or constructs an optimisation score.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import E_CHARGE_C
from .particles import ParticleDump, select_hot_electrons, weighted_percentile
from .soft50 import effective_sample_size
from .transverse import summarize_transverse_metrics


BEAM_EVOLUTION_SCHEMA = "beam_evolution_v1"
QUALITY_COLUMNS = (
    "emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad",
    "theta_x_std_mrad", "theta_y_std_mrad",
    "energy_spread_rms_MeV", "energy_relative_spread_rms",
)
FRAME_COLUMNS = (
    "schema_version", "case_name", "species_scope", "iteration", "time_fs",
    "z_frame_um", "coordinate_source", "energy_threshold_MeV", "forward_only",
    "min_effective_particles", "population_status", "quality_status", "read_error",
    "n_macroparticles", "n_invalid_macroparticles", "n_effective", "charge_pC",
    "energy_mean_MeV", "energy_spread_rms_MeV", "energy_relative_spread_rms",
    "energy_p10_MeV", "energy_p50_MeV", "energy_p90_MeV", "energy_p95_MeV",
    "theta_x_mean_mrad", "theta_y_mean_mrad", "theta_x_std_mrad", "theta_y_std_mrad",
    "theta_x_axis_rms_mrad", "theta_y_axis_rms_mrad",
    "emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad",
    "x_mean_um", "y_mean_um", "z_centroid_um", "x_rms_um", "y_rms_um",
    "radius_rms_um", "x_ux_cov_um", "y_uy_cov_um",
)


@dataclass(frozen=True)
class BeamEvolutionConfig:
    energy_threshold_mev: float = 50.0
    min_effective_particles: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_threshold_mev) or self.energy_threshold_mev < 0:
            raise ValueError("energy threshold must be finite and non-negative")
        if not math.isfinite(self.min_effective_particles) or self.min_effective_particles < 3:
            raise ValueError("min_effective_particles must be finite and >= 3")


def _frame_template(
    *, iteration: int, time_fs: float, species_scope: str, z_frame_um: float,
    coordinate_source: str, config: BeamEvolutionConfig,
) -> dict[str, Any]:
    row = dict.fromkeys(FRAME_COLUMNS, float("nan"))
    row.update(
        schema_version=BEAM_EVOLUTION_SCHEMA, case_name="", species_scope=species_scope,
        iteration=int(iteration), time_fs=float(time_fs), z_frame_um=float(z_frame_um),
        coordinate_source=coordinate_source, energy_threshold_MeV=config.energy_threshold_mev,
        forward_only=True, min_effective_particles=config.min_effective_particles,
        population_status="unavailable", quality_status="unavailable", read_error="",
    )
    return row


def unavailable_beam_frame(
    *, iteration: int, time_fs: float, species_scope: str, z_frame_um: float,
    coordinate_source: str, error: str, config: BeamEvolutionConfig | None = None,
) -> dict[str, Any]:
    row = _frame_template(
        iteration=iteration, time_fs=time_fs, species_scope=species_scope,
        z_frame_um=z_frame_um, coordinate_source=coordinate_source,
        config=config or BeamEvolutionConfig(),
    )
    row["read_error"] = error
    return row


def summarize_beam_frame(
    dump: ParticleDump, *, species_scope: str, z_frame_um: float,
    coordinate_source: str = "guiding_exact", config: BeamEvolutionConfig | None = None,
) -> dict[str, Any]:
    """Reduce all forward electrons above the hard threshold, without acceptance cuts.

    Covariances and variances use sum(w), without a sample Bessel correction.
    Low-statistics moments are retained but flagged and excluded from quality
    integrals. The minimum of three is a rank/statistics floor, not convergence.
    """
    cfg = config or BeamEvolutionConfig()
    row = _frame_template(
        iteration=dump.iteration, time_fs=dump.time_fs, species_scope=species_scope,
        z_frame_um=z_frame_um, coordinate_source=coordinate_source, config=cfg,
    )
    arrays = [np.asarray(getattr(dump, name), dtype=float) for name in
              ("x_m", "y_m", "z_m", "ux", "uy", "uz", "w")]
    if any(a.ndim != 1 or a.shape != arrays[-1].shape for a in arrays):
        raise ValueError("Particle arrays must be one-dimensional and have equal lengths")
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays]) & (arrays[-1] > 0)
    # Zero-weight particles are inactive; other invalid records are unknown data.
    invalid = (arrays[-1] != 0) & ~valid
    selected = select_hot_electrons(
        dump, hot_energy_mev=cfg.energy_threshold_mev, forward_only=True,
        longitudinal="z", exit_window_mm=None,
    ) & valid
    w = arrays[-1][selected]
    count = int(np.count_nonzero(selected))
    n_eff = effective_sample_size(w)
    row.update(
        n_macroparticles=count, n_invalid_macroparticles=int(np.count_nonzero(invalid)),
        n_effective=n_eff, charge_pC=float(np.sum(w)) * E_CHARGE_C * 1e12,
        population_status="invalid_particle_data" if np.any(invalid) else ("ok" if count else "empty"),
        quality_status="empty" if not count else "insufficient_statistics",
    )
    if not count:
        return row

    def mean(a: np.ndarray) -> float:
        return float(np.average(a, weights=w))

    def std(a: np.ndarray) -> float:
        return float(np.sqrt(mean((a - mean(a)) ** 2)))

    energy = np.asarray(dump.kinetic_energy_mev, dtype=float)[selected]
    energy_mean = mean(energy)
    spread = std(energy)
    row.update(energy_mean_MeV=energy_mean, energy_spread_rms_MeV=spread,
               energy_relative_spread_rms=spread / energy_mean if energy_mean > 0 else float("nan"))
    for percentile in (10, 50, 90, 95):
        row[f"energy_p{percentile}_MeV"] = weighted_percentile(energy, w, percentile)
    transverse = summarize_transverse_metrics(dump, mask=selected, longitudinal="z")
    for key in ("emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad", "x_rms_um", "y_rms_um"):
        row[key] = transverse[key]
    x, y, z, ux, uy, uz = (a[selected] for a in arrays[:-1])
    for axis, pos, momentum in (("x", x, ux), ("y", y, uy)):
        angle = np.arctan2(momentum, uz)
        row[f"theta_{axis}_mean_mrad"] = mean(angle) * 1e3
        row[f"theta_{axis}_std_mrad"] = std(angle) * 1e3
        row[f"theta_{axis}_axis_rms_mrad"] = transverse[f"theta_{axis}_rms_mrad"]
        row[f"{axis}_mean_um"] = mean(pos) * 1e6
        row[f"{axis}_u{axis}_cov_um"] = mean((pos - mean(pos)) * (momentum - mean(momentum))) * 1e6
    row["z_centroid_um"] = mean(z) * 1e6
    row["radius_rms_um"] = float(np.sqrt(mean(x * x + y * y))) * 1e6
    if count >= 3 and n_eff >= cfg.min_effective_particles and not np.any(invalid):
        row["quality_status"] = "ok"
    return row


def _number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def resolve_frame_coordinate(
    guiding_rows: Sequence[dict[str, Any]], *, iteration: int, time_fs: float,
    max_coordinate_gap_um: float | None = None,
) -> tuple[float, str]:
    """Exact guiding association, or bracketed interpolation in physical time.

    No extrapolation, nearest-iteration replacement, c*t guess or laser-peak
    fallback. The value is the absolute last mesh sample z_max_um.
    """
    if max_coordinate_gap_um is not None and (
        not math.isfinite(max_coordinate_gap_um) or max_coordinate_gap_um <= 0
    ):
        raise ValueError("max_coordinate_gap_um must be finite and positive")
    exact = [r for r in guiding_rows if int(r["iteration"]) == int(iteration)]
    if len(exact) > 1:
        raise ValueError("Duplicate guiding iteration")
    if exact:
        row = exact[0]
        t = _number(row, "time_fs")
        if math.isfinite(t) and math.isfinite(time_fs) and not math.isclose(t, time_fs, rel_tol=1e-9, abs_tol=1e-3):
            raise ValueError("Particle and guiding timestamps disagree")
        z = _number(row, "z_max_um")
        return (z, "guiding_exact") if math.isfinite(z) else (float("nan"), "unavailable")
    if not math.isfinite(time_fs):
        return float("nan"), "unavailable"
    timed = sorted((r for r in guiding_rows if math.isfinite(_number(r, "time_fs"))),
                   key=lambda r: _number(r, "time_fs"))
    for left, right in zip(timed, timed[1:]):
        t0, t1 = _number(left, "time_fs"), _number(right, "time_fs")
        if t0 <= time_fs <= t1 and t1 > t0:
            z0, z1 = _number(left, "z_max_um"), _number(right, "z_max_um")
            if not (math.isfinite(z0) and math.isfinite(z1)) or z1 < z0:
                return float("nan"), "unavailable"
            if max_coordinate_gap_um is not None and z1 - z0 > max_coordinate_gap_um:
                return float("nan"), "unavailable"
            return z0 + (z1 - z0) * (time_fs - t0) / (t1 - t0), "guiding_time_interpolated"
    return float("nan"), "unavailable"


def _clipped_trapezoid(
    z0: float, z1: float, y0: float, y1: float, start_um: float, end_um: float,
) -> tuple[float, float]:
    lo, hi = max(z0, start_um), min(z1, end_um)
    if hi <= lo:
        return 0.0, 0.0
    a = y0 + (y1 - y0) * (lo - z0) / (z1 - z0)
    b = y0 + (y1 - y0) * (hi - z0) / (z1 - z0)
    return (hi - lo) * (a + b) / 2, hi - lo


def summarize_beam_evolution(
    rows: Sequence[dict[str, Any]], *, plasma_start_um: float, plasma_end_um: float,
    max_gap_um: float | None = None, exit_iteration: int | None = None,
) -> dict[str, Any]:
    """Integrate a single species across the full absolute plasma interval.

    Integrate piecewise-linear Q and Q*A, clipping bracketed segments at bounds.
    Never bridge a missing/invalid row or extrapolate. Canonical integrals/means
    are NaN if coverage is incomplete; explicitly named observed values retain
    partial information. Each quality observable records its own coverage.
    """
    start, end = float(plasma_start_um), float(plasma_end_um)
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise ValueError("Plasma bounds must be finite with end > start")
    if not rows:
        raise ValueError("No beam rows to aggregate")
    ordered = sorted(rows, key=lambda r: int(r["iteration"]))
    for key in ("schema_version", "case_name", "species_scope", "energy_threshold_MeV",
                "forward_only", "min_effective_particles"):
        if any(r[key] != ordered[0][key] for r in ordered):
            raise ValueError(f"Cannot aggregate different {key}")
    if len({int(r["iteration"]) for r in ordered}) != len(ordered):
        raise ValueError("Duplicate particle iteration")
    coordinates = [_number(r, "z_frame_um") for r in ordered]
    finite_z = [z for z in coordinates if math.isfinite(z)]
    if any(b < a for a, b in zip(finite_z, finite_z[1:])):
        raise ValueError("Frame coordinate must be non-decreasing in iteration order")
    separations = [b - a for a, b in zip(coordinates, coordinates[1:])
                   if math.isfinite(a) and math.isfinite(b) and b > a]
    inferred_gap = 1.5 * float(np.median(separations)) if separations else float("nan")
    gap_limit = inferred_gap if max_gap_um is None else float(max_gap_um)
    if max_gap_um is not None and (not math.isfinite(gap_limit) or gap_limit <= 0):
        raise ValueError("max_gap_um must be finite and positive")
    length = end - start
    charge_integral = covered = 0.0
    totals = {key: [0.0, 0.0, 0.0] for key in QUALITY_COLUMNS}  # QA integral, Q integral, length
    skipped_gaps = 0
    for left, right in zip(ordered, ordered[1:]):
        z0, z1 = _number(left, "z_frame_um"), _number(right, "z_frame_um")
        if not (math.isfinite(z0) and math.isfinite(z1)) or z1 <= z0:
            continue
        if min(z1, end) <= max(z0, start):
            continue
        if z1 - z0 > gap_limit:
            skipped_gaps += 1
            continue
        q0, q1 = _number(left, "charge_pC"), _number(right, "charge_pC")
        if any(r["population_status"] not in ("ok", "empty") for r in (left, right)):
            continue
        if not (math.isfinite(q0) and math.isfinite(q1) and q0 >= 0 and q1 >= 0):
            continue
        iq, segment_length = _clipped_trapezoid(z0, z1, q0, q1, start, end)
        charge_integral += iq
        covered += segment_length
        for key in QUALITY_COLUMNS:
            products = []
            for r, q in ((left, q0), (right, q1)):
                a = _number(r, key)
                products.append(0.0 if q == 0 else
                                q * a if r["quality_status"] == "ok" and math.isfinite(a) else float("nan"))
            if all(math.isfinite(p) for p in products):
                numerator, _ = _clipped_trapezoid(z0, z1, *products, start, end)
                totals[key][0] += numerator
                totals[key][1] += iq
                totals[key][2] += segment_length

    complete = math.isclose(covered, length, rel_tol=1e-10, abs_tol=1e-6)
    summary: dict[str, Any] = {
        "schema_version": BEAM_EVOLUTION_SCHEMA, "case_name": ordered[0]["case_name"],
        "species_scope": ordered[0]["species_scope"],
        "energy_threshold_MeV": ordered[0]["energy_threshold_MeV"], "forward_only": True,
        "min_effective_particles": ordered[0]["min_effective_particles"],
        "coordinate_reference": "guiding_z_max_um_absolute", "plasma_start_um": start,
        "plasma_end_um": end, "plasma_length_um": length, "n_frames": len(ordered),
        "integration_method": "clipped_trapezoid_Q_and_QA",
        "max_gap_um": gap_limit, "max_gap_policy": "explicit" if max_gap_um is not None else "1.5_median_frame_spacing",
        "n_skipped_gap_segments": skipped_gaps,
        "integration_status": "complete" if complete else "partial" if covered > 0 else "no_coverage",
        "covered_length_um": covered, "coverage_fraction": min(covered / length, 1.0),
        "charge_distance_observed_pC_um": charge_integral if covered > 0 else float("nan"),
        "charge_distance_pC_um": charge_integral if complete else float("nan"),
        "exit_iteration": exit_iteration if exit_iteration is not None else "",
        "exit_snapshot_status": "not_requested", "exit_z_frame_um": float("nan"),
        "exit_quality_status": "unavailable", "charge_exit_snapshot_pC": float("nan"),
    }
    for key, (numerator, denominator, support) in totals.items():
        mean = numerator / denominator if denominator > 0 else float("nan")
        quality_complete = math.isclose(support, length, rel_tol=1e-10, abs_tol=1e-6)
        summary[f"{key}_charge_weighted_observed"] = mean
        summary[f"{key}_charge_weighted_mean"] = mean if complete and quality_complete else float("nan")
        summary[f"{key}_coverage_fraction"] = min(support / length, 1.0)
        summary[f"{key}_charge_distance_support_pC_um"] = denominator
        summary[f"{key}_exit_snapshot"] = float("nan")
    if exit_iteration is not None:
        matches = [r for r in ordered if int(r["iteration"]) == int(exit_iteration)]
        summary["exit_snapshot_status"] = "missing"
        if matches:
            r = matches[0]
            summary["exit_snapshot_status"] = r["population_status"]
            summary["exit_z_frame_um"] = _number(r, "z_frame_um")
            summary["exit_quality_status"] = r["quality_status"]
            if r["population_status"] in ("ok", "empty"):
                summary["charge_exit_snapshot_pC"] = r["charge_pC"]
                for key in QUALITY_COLUMNS:
                    summary[f"{key}_exit_snapshot"] = r[key]
    return summary


def write_beam_csv(rows: Sequence[dict[str, Any]], path: str | Path, *, frames: bool = False) -> Path:
    if not rows:
        raise ValueError("No beam rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(FRAME_COLUMNS) if frames else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
