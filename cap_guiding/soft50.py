from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import E_CHARGE_C


SOFT50_SCHEMA_VERSION = "soft50_v2"
SOFT50_CURVE_COLUMNS = [
    "case_id",
    "case_name",
    "species_scope",
    "iteration",
    "selection_mode",
    "selected_particle_iteration",
    "soft50_schema_version",
    "soft50_energy_low_MeV",
    "soft50_energy_target_MeV",
    "soft50_reliability_floor",
    "soft50_effective_count_reference",
    "soft50_status",
    "n_macroparticles_soft50",
    "weight_soft50",
    "charge_soft50_pC",
    "n_effective_soft50",
    "reliability_soft50",
    "energy_mean_soft50_MeV",
    "energy_spread_rms_soft50_MeV",
    "energy_relative_spread_rms_soft50",
    "energy_p10_soft50_MeV",
    "energy_p50_soft50_MeV",
    "energy_p90_soft50_MeV",
    "energy_p95_soft50_MeV",
    "theta_r_p90_soft50_mrad",
    "theta_r_p95_soft50_mrad",
    "emitn_x_soft50_um_rad",
    "emitn_y_soft50_um_rad",
    "emitn_xy_soft50_um_rad",
    "charge_Ege50MeV_pC",
    "n_macroparticles_Ege50MeV",
    "longitudinal_coordinate",
    "forward_only",
    "exit_window_mm",
]


@dataclass(frozen=True)
class Soft50Config:
    """Versioned particle-level acceptance around a target kinetic energy."""

    energy_low_mev: float = 10.0
    energy_target_mev: float = 50.0
    reliability_floor: float = 0.05
    effective_count_reference: float = 200.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.energy_low_mev,
                self.energy_target_mev,
                self.reliability_floor,
                self.effective_count_reference,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Soft50Config values must be finite")
        if self.energy_low_mev < 0.0:
            raise ValueError("energy_low_mev must be non-negative")
        if self.energy_target_mev <= self.energy_low_mev:
            raise ValueError("energy_target_mev must be greater than energy_low_mev")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be within [0, 1]")
        if self.effective_count_reference <= 0.0:
            raise ValueError("effective_count_reference must be positive")


def smooth_energy_acceptance(
    energy_mev: np.ndarray,
    *,
    energy_low_mev: float,
    energy_target_mev: float,
) -> np.ndarray:
    """Return cubic smoothstep acceptance, zero below low and one at target."""

    low = float(energy_low_mev)
    target = float(energy_target_mev)
    if not np.isfinite(low) or not np.isfinite(target) or target <= low:
        raise ValueError("energy acceptance requires finite target > low")

    energy = np.asarray(energy_mev, dtype=float)
    u = np.clip((energy - low) / (target - low), 0.0, 1.0)
    accepted = 3.0 * u * u - 2.0 * u * u * u
    accepted[~np.isfinite(energy)] = 0.0
    return accepted


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return 0.0
    selected = weights[valid]
    denominator = float(np.sum(selected * selected))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(selected) ** 2 / denominator)


def soft_reliability(
    n_effective: float,
    *,
    reliability_floor: float,
    effective_count_reference: float,
) -> float:
    n_eff = max(float(n_effective), 0.0)
    floor = float(reliability_floor)
    reference = float(effective_count_reference)
    if not np.isfinite(n_eff) or not np.isfinite(floor) or not np.isfinite(reference):
        raise ValueError("reliability inputs must be finite")
    if not 0.0 <= floor <= 1.0 or reference <= 0.0:
        raise ValueError("invalid reliability floor/reference")
    return float(floor + (1.0 - floor) * (1.0 - np.exp(-n_eff / reference)))


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError("percentile must be within [0, 100]")

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(percentile) / 100.0 * cumulative[-1]
    return float(values[np.searchsorted(cumulative, target, side="left")])


def _longitudinal_arrays(dump: Any, longitudinal: str) -> tuple[np.ndarray, np.ndarray]:
    if longitudinal == "z":
        return np.asarray(dump.z_m), np.asarray(dump.uz)
    if longitudinal == "x":
        return np.asarray(dump.x_m), np.asarray(dump.ux)
    if longitudinal == "y":
        return np.asarray(dump.y_m), np.asarray(dump.uy)
    raise ValueError("longitudinal must be one of: x, y, z")


def _base_particle_mask(
    dump: Any,
    *,
    longitudinal: str,
    forward_only: bool,
    exit_window_mm: float | None,
) -> np.ndarray:
    energy = np.asarray(dump.kinetic_energy_mev, dtype=float)
    weights = np.asarray(dump.w, dtype=float)
    q_long, u_long = _longitudinal_arrays(dump, longitudinal)
    mask = (
        np.isfinite(energy)
        & np.isfinite(weights)
        & (weights > 0.0)
        & np.isfinite(q_long)
        & np.isfinite(u_long)
    )
    if forward_only:
        mask &= u_long > 0.0
    if exit_window_mm is not None:
        window_m = float(exit_window_mm) * 1.0e-3
        if not np.isfinite(window_m) or window_m <= 0.0:
            raise ValueError("exit_window_mm must be finite and positive")
        if np.any(mask):
            qmax = float(np.max(q_long[mask]))
            mask &= q_long >= qmax - window_m
    return mask


def _weighted_covariance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    mean_a = float(np.average(a, weights=weights))
    mean_b = float(np.average(b, weights=weights))
    return float(np.average((a - mean_a) * (b - mean_b), weights=weights))


def _normalized_emittance_um_rad(
    position_m: np.ndarray,
    normalized_momentum: np.ndarray,
    weights: np.ndarray,
) -> float:
    q_var = _weighted_covariance(position_m, position_m, weights)
    u_var = _weighted_covariance(normalized_momentum, normalized_momentum, weights)
    qu_cov = _weighted_covariance(position_m, normalized_momentum, weights)
    determinant = q_var * u_var - qu_cov * qu_cov
    if determinant < 0.0 and abs(determinant) <= 1.0e-24:
        determinant = 0.0
    return float(np.sqrt(max(determinant, 0.0)) * 1.0e6)


def summarize_soft50_metrics(
    dump: Any,
    *,
    config: Soft50Config | None = None,
    longitudinal: str = "z",
    forward_only: bool = True,
    exit_window_mm: float | None = None,
) -> dict[str, Any]:
    cfg = config or Soft50Config()
    energy = np.asarray(dump.kinetic_energy_mev, dtype=float)
    physical_weights = np.asarray(dump.w, dtype=float)
    base = _base_particle_mask(
        dump,
        longitudinal=longitudinal,
        forward_only=forward_only,
        exit_window_mm=exit_window_mm,
    )
    acceptance = smooth_energy_acceptance(
        energy,
        energy_low_mev=cfg.energy_low_mev,
        energy_target_mev=cfg.energy_target_mev,
    )
    soft_weights = np.where(base, physical_weights * acceptance, 0.0)
    soft = soft_weights > 0.0
    hard = base & (energy >= cfg.energy_target_mev)

    weight_soft = float(np.sum(soft_weights))
    hard_weight = float(np.sum(physical_weights[hard])) if np.any(hard) else 0.0
    n_effective = effective_sample_size(soft_weights)
    reliability = soft_reliability(
        n_effective,
        reliability_floor=cfg.reliability_floor,
        effective_count_reference=cfg.effective_count_reference,
    )

    row: dict[str, Any] = {
        "soft50_schema_version": SOFT50_SCHEMA_VERSION,
        "soft50_energy_low_MeV": float(cfg.energy_low_mev),
        "soft50_energy_target_MeV": float(cfg.energy_target_mev),
        "soft50_reliability_floor": float(cfg.reliability_floor),
        "soft50_effective_count_reference": float(cfg.effective_count_reference),
        "soft50_status": "ok" if np.any(soft) else "no_accepted_particles",
        "n_macroparticles_soft50": int(np.count_nonzero(soft)),
        "weight_soft50": weight_soft,
        "charge_soft50_pC": weight_soft * E_CHARGE_C / 1.0e-12,
        "n_effective_soft50": n_effective,
        "reliability_soft50": reliability,
        "charge_Ege50MeV_pC": hard_weight * E_CHARGE_C / 1.0e-12,
        "n_macroparticles_Ege50MeV": int(np.count_nonzero(hard)),
    }

    if not np.any(soft):
        row.update(
            {
                "energy_mean_soft50_MeV": float("nan"),
                "energy_spread_rms_soft50_MeV": float("nan"),
                "energy_relative_spread_rms_soft50": float("nan"),
                "energy_p10_soft50_MeV": float("nan"),
                "energy_p50_soft50_MeV": float("nan"),
                "energy_p90_soft50_MeV": float("nan"),
                "energy_p95_soft50_MeV": float("nan"),
                "theta_r_p90_soft50_mrad": float("nan"),
                "theta_r_p95_soft50_mrad": float("nan"),
                "emitn_x_soft50_um_rad": float("nan"),
                "emitn_y_soft50_um_rad": float("nan"),
                "emitn_xy_soft50_um_rad": float("nan"),
            }
        )
        return row

    selected_energy = energy[soft]
    selected_weights = soft_weights[soft]
    energy_mean = float(np.average(selected_energy, weights=selected_weights))
    energy_variance = float(
        np.average((selected_energy - energy_mean) ** 2, weights=selected_weights)
    )
    energy_spread_rms = math.sqrt(max(energy_variance, 0.0))

    row.update(
        {
            "energy_mean_soft50_MeV": energy_mean,
            "energy_spread_rms_soft50_MeV": energy_spread_rms,
            "energy_relative_spread_rms_soft50": (
                energy_spread_rms / energy_mean
                if energy_mean > 0.0
                else float("nan")
            ),
            "energy_p10_soft50_MeV": weighted_percentile(
                energy, soft_weights, 10.0
            ),
            "energy_p50_soft50_MeV": weighted_percentile(energy, soft_weights, 50.0),
            "energy_p90_soft50_MeV": weighted_percentile(energy, soft_weights, 90.0),
            "energy_p95_soft50_MeV": weighted_percentile(energy, soft_weights, 95.0),
        }
    )

    if longitudinal != "z":
        row.update(
            {
                "theta_r_p90_soft50_mrad": float("nan"),
                "theta_r_p95_soft50_mrad": float("nan"),
                "emitn_x_soft50_um_rad": float("nan"),
                "emitn_y_soft50_um_rad": float("nan"),
                "emitn_xy_soft50_um_rad": float("nan"),
            }
        )
        return row

    valid_transverse = (
        soft
        & np.isfinite(dump.x_m)
        & np.isfinite(dump.y_m)
        & np.isfinite(dump.ux)
        & np.isfinite(dump.uy)
        & np.isfinite(dump.uz)
    )
    if not np.any(valid_transverse):
        row["soft50_status"] = "no_transverse_particles"
        row.update(
            {
                "theta_r_p90_soft50_mrad": float("nan"),
                "theta_r_p95_soft50_mrad": float("nan"),
                "emitn_x_soft50_um_rad": float("nan"),
                "emitn_y_soft50_um_rad": float("nan"),
                "emitn_xy_soft50_um_rad": float("nan"),
            }
        )
        return row

    weights = soft_weights[valid_transverse]
    x = np.asarray(dump.x_m[valid_transverse], dtype=float)
    y = np.asarray(dump.y_m[valid_transverse], dtype=float)
    ux = np.asarray(dump.ux[valid_transverse], dtype=float)
    uy = np.asarray(dump.uy[valid_transverse], dtype=float)
    uz = np.asarray(dump.uz[valid_transverse], dtype=float)
    theta_r_mrad = np.sqrt(np.arctan2(ux, uz) ** 2 + np.arctan2(uy, uz) ** 2) * 1.0e3
    emit_x = _normalized_emittance_um_rad(x, ux, weights)
    emit_y = _normalized_emittance_um_rad(y, uy, weights)
    row.update(
        {
            "theta_r_p90_soft50_mrad": weighted_percentile(
                theta_r_mrad, weights, 90.0
            ),
            "theta_r_p95_soft50_mrad": weighted_percentile(
                theta_r_mrad, weights, 95.0
            ),
            "emitn_x_soft50_um_rad": emit_x,
            "emitn_y_soft50_um_rad": emit_y,
            "emitn_xy_soft50_um_rad": float(
                np.sqrt(max(emit_x, 0.0) * max(emit_y, 0.0))
            ),
        }
    )
    return row


def summarize_soft50_curve(
    dump: Any,
    *,
    energy_low_values_mev: Sequence[float],
    energy_target_mev: float = 50.0,
    reliability_floor: float = 0.05,
    effective_count_reference: float = 200.0,
    longitudinal: str = "z",
    forward_only: bool = True,
    exit_window_mm: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    values = sorted({float(value) for value in energy_low_values_mev})
    if not values:
        raise ValueError("energy_low_values_mev must not be empty")
    rows: list[dict[str, Any]] = []
    for energy_low in values:
        metrics = summarize_soft50_metrics(
            dump,
            config=Soft50Config(
                energy_low_mev=energy_low,
                energy_target_mev=energy_target_mev,
                reliability_floor=reliability_floor,
                effective_count_reference=effective_count_reference,
            ),
            longitudinal=longitudinal,
            forward_only=forward_only,
            exit_window_mm=exit_window_mm,
        )
        rows.append(
            {
                **(metadata or {}),
                **metrics,
                "iteration": int(dump.iteration),
                "longitudinal_coordinate": longitudinal,
                "forward_only": bool(forward_only),
                "exit_window_mm": (
                    float(exit_window_mm) if exit_window_mm is not None else ""
                ),
            }
        )
    return rows


def write_soft50_curve_csv(
    rows: Iterable[dict[str, Any]], path: str | Path
) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("No soft50 curve rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOFT50_CURVE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SOFT50_CURVE_COLUMNS})
    return path
