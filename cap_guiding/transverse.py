from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MRAD_PER_RAD = 1.0e3
UM_PER_M = 1.0e6
MM_MRAD_PER_M_RAD = 1.0e6


TRANSVERSE_OUTPUT_COLUMNS = [
    "transverse_status",
    "n_macroparticles_transverse",
    "weight_transverse",
    "theta_x_rms_mrad",
    "theta_y_rms_mrad",
    "theta_rms_mrad",
    "theta_x_p95_mrad",
    "theta_y_p95_mrad",
    "theta_r_p95_mrad",
    "x_rms_um",
    "y_rms_um",
    "x_p95_um",
    "y_p95_um",
    "emit_x_norm_mm_mrad",
    "emit_y_norm_mm_mrad",
    "emit_geom_norm_mm_mrad",
    "transverse_theta_rms_component",
    "transverse_theta_p95_component",
    "transverse_emit_component",
    "beam_transverse_quality_score",
]


@dataclass(frozen=True)
class TransverseQualityConfig:
    """Configuration for the separate transverse-quality proxy.

    The score is deliberately independent from beamlike_score. It rewards low
    angular spread/tails and low normalized transverse emittance for the already
    selected hot/forward electron population.
    """

    score_scale: float = 1000.0
    theta_rms_ref_mrad: float = 5.0
    theta_p95_ref_mrad: float = 15.0
    emit_ref_mm_mrad: float = 2.0


def _empty_transverse_metrics(status: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "transverse_status": status,
        "n_macroparticles_transverse": 0,
        "weight_transverse": 0.0,
        "beam_transverse_quality_score": 0.0,
        "transverse_theta_rms_component": 0.0,
        "transverse_theta_p95_component": 0.0,
        "transverse_emit_component": 0.0,
    }

    for column in TRANSVERSE_OUTPUT_COLUMNS:
        if column not in out:
            out[column] = float("nan")

    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(values * values, weights=weights)))


def _weighted_centered_rms(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    centered = values - mean
    return _weighted_rms(centered, weights)


def _weighted_covariance(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
) -> float:
    a0 = a - _weighted_mean(a, weights)
    b0 = b - _weighted_mean(b, weights)
    return float(np.average(a0 * b0, weights=weights))


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")

    v = np.asarray(values[finite], dtype=float)
    w = np.asarray(weights[finite], dtype=float)

    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cumulative = np.cumsum(w)
    target = float(percentile) / 100.0 * cumulative[-1]
    return float(v[np.searchsorted(cumulative, target, side="left")])


def _normalized_emittance_m_rad(
    q_m: np.ndarray,
    u_transverse: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return centered normalized RMS emittance in m rad.

    ux, uy, uz are WarpX normalized momenta gamma*v/c. Therefore the normalized
    emittance computed from position and u_transverse has units of length; the
    conventional display unit is mm mrad.
    """
    q_var = _weighted_covariance(q_m, q_m, weights)
    u_var = _weighted_covariance(u_transverse, u_transverse, weights)
    qu_cov = _weighted_covariance(q_m, u_transverse, weights)

    determinant = q_var * u_var - qu_cov * qu_cov
    if determinant < 0.0 and abs(determinant) <= 1.0e-24:
        determinant = 0.0

    return float(np.sqrt(max(determinant, 0.0)))


def _inverse_scale_component(value: float, reference: float) -> float:
    if not np.isfinite(value) or reference <= 0.0:
        return 0.0
    value = max(float(value), 0.0)
    return float(1.0 / (1.0 + value / reference))


def beam_transverse_quality_score(
    *,
    theta_rms_mrad: float,
    theta_r_p95_mrad: float,
    emit_x_norm_mm_mrad: float,
    emit_y_norm_mm_mrad: float,
    config: TransverseQualityConfig | None = None,
) -> dict[str, float]:
    cfg = config or TransverseQualityConfig()

    theta_rms_component = _inverse_scale_component(
        theta_rms_mrad,
        cfg.theta_rms_ref_mrad,
    )
    theta_p95_component = _inverse_scale_component(
        theta_r_p95_mrad,
        cfg.theta_p95_ref_mrad,
    )

    if np.isfinite(emit_x_norm_mm_mrad) and np.isfinite(emit_y_norm_mm_mrad):
        emit_geom = float(
            np.sqrt(max(emit_x_norm_mm_mrad, 0.0) * max(emit_y_norm_mm_mrad, 0.0))
        )
        emit_component = _inverse_scale_component(emit_geom, cfg.emit_ref_mm_mrad)
    else:
        emit_component = 0.0

    score = cfg.score_scale * theta_rms_component * theta_p95_component * emit_component

    return {
        "transverse_theta_rms_component": float(theta_rms_component),
        "transverse_theta_p95_component": float(theta_p95_component),
        "transverse_emit_component": float(emit_component),
        "beam_transverse_quality_score": float(score),
    }


def summarize_transverse_metrics(
    dump: Any,
    *,
    mask: np.ndarray,
    longitudinal: str = "z",
    config: TransverseQualityConfig | None = None,
) -> dict[str, Any]:
    """Compute transverse metrics for the selected beam population.

    Fase 1 only assigns the requested x/y transverse meaning when the campaign
    longitudinal axis is z. Other axes return explicit NaNs rather than inventing
    a renamed coordinate convention.
    """
    if longitudinal != "z":
        return _empty_transverse_metrics(
            status=f"unsupported_longitudinal_axis:{longitudinal}"
        )

    selection = np.asarray(mask, dtype=bool)
    if selection.shape != np.asarray(dump.w).shape:
        raise ValueError(
            "Transverse mask shape mismatch: "
            f"mask.shape={selection.shape}, w.shape={np.asarray(dump.w).shape}"
        )

    valid = (
        selection
        & np.isfinite(dump.x_m)
        & np.isfinite(dump.y_m)
        & np.isfinite(dump.ux)
        & np.isfinite(dump.uy)
        & np.isfinite(dump.uz)
        & np.isfinite(dump.w)
        & (dump.w > 0.0)
    )

    if not np.any(valid):
        return _empty_transverse_metrics(status="no_selected_particles")

    x = np.asarray(dump.x_m[valid], dtype=float)
    y = np.asarray(dump.y_m[valid], dtype=float)
    ux = np.asarray(dump.ux[valid], dtype=float)
    uy = np.asarray(dump.uy[valid], dtype=float)
    uz = np.asarray(dump.uz[valid], dtype=float)
    w = np.asarray(dump.w[valid], dtype=float)

    theta_x = np.arctan2(ux, uz)
    theta_y = np.arctan2(uy, uz)
    theta_r = np.sqrt(theta_x * theta_x + theta_y * theta_y)

    theta_x_rms_mrad = _weighted_rms(theta_x, w) * MRAD_PER_RAD
    theta_y_rms_mrad = _weighted_rms(theta_y, w) * MRAD_PER_RAD
    theta_rms_mrad = _weighted_rms(theta_r, w) * MRAD_PER_RAD

    theta_x_p95_mrad = _weighted_percentile(np.abs(theta_x), w, 95.0) * MRAD_PER_RAD
    theta_y_p95_mrad = _weighted_percentile(np.abs(theta_y), w, 95.0) * MRAD_PER_RAD
    theta_r_p95_mrad = _weighted_percentile(theta_r, w, 95.0) * MRAD_PER_RAD

    x_rms_um = _weighted_centered_rms(x, w) * UM_PER_M
    y_rms_um = _weighted_centered_rms(y, w) * UM_PER_M
    x_mean = _weighted_mean(x, w)
    y_mean = _weighted_mean(y, w)
    x_p95_um = _weighted_percentile(np.abs(x - x_mean), w, 95.0) * UM_PER_M
    y_p95_um = _weighted_percentile(np.abs(y - y_mean), w, 95.0) * UM_PER_M

    emit_x_norm_mm_mrad = _normalized_emittance_m_rad(x, ux, w) * MM_MRAD_PER_M_RAD
    emit_y_norm_mm_mrad = _normalized_emittance_m_rad(y, uy, w) * MM_MRAD_PER_M_RAD
    emit_geom_norm_mm_mrad = float(
        np.sqrt(max(emit_x_norm_mm_mrad, 0.0) * max(emit_y_norm_mm_mrad, 0.0))
    )

    out: dict[str, Any] = {
        "transverse_status": "ok",
        "n_macroparticles_transverse": int(np.count_nonzero(valid)),
        "weight_transverse": float(np.sum(w)),
        "theta_x_rms_mrad": float(theta_x_rms_mrad),
        "theta_y_rms_mrad": float(theta_y_rms_mrad),
        "theta_rms_mrad": float(theta_rms_mrad),
        "theta_x_p95_mrad": float(theta_x_p95_mrad),
        "theta_y_p95_mrad": float(theta_y_p95_mrad),
        "theta_r_p95_mrad": float(theta_r_p95_mrad),
        "x_rms_um": float(x_rms_um),
        "y_rms_um": float(y_rms_um),
        "x_p95_um": float(x_p95_um),
        "y_p95_um": float(y_p95_um),
        "emit_x_norm_mm_mrad": float(emit_x_norm_mm_mrad),
        "emit_y_norm_mm_mrad": float(emit_y_norm_mm_mrad),
        "emit_geom_norm_mm_mrad": float(emit_geom_norm_mm_mrad),
    }
    out.update(
        beam_transverse_quality_score(
            theta_rms_mrad=out["theta_rms_mrad"],
            theta_r_p95_mrad=out["theta_r_p95_mrad"],
            emit_x_norm_mm_mrad=out["emit_x_norm_mm_mrad"],
            emit_y_norm_mm_mrad=out["emit_y_norm_mm_mrad"],
            config=config,
        )
    )

    return out
