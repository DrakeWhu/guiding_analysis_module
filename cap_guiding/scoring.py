from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .case_metadata import infer_plateau_window_mm_from_text


REQUIRED_SCORE_COLUMNS = [
    "iteration",
    "propagation_mm",
    "a0_peak",
    "waist_um",
]


@dataclass(frozen=True)
class ScoreConfig:
    entry_window_mm: float = 1.0
    exit_before_mm: float = 1.0
    exit_after_mm: float = 2.0
    a0_target: float = 1.5
    a0_component_cap: float = 1.25
    waist_growth_sigma: float = 0.75
    waist_jitter_sigma: float = 0.25
    weight_a0_exit: float = 0.50
    weight_a0_retention: float = 0.20
    weight_waist_growth: float = 0.20
    weight_waist_stability: float = 0.10


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _finite_median(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _finite_max(values: pd.Series) -> tuple[float, int | None]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if np.all(~np.isfinite(arr)):
        return float("nan"), None
    idx = int(np.nanargmax(arr))
    return float(arr[idx]), idx


def _clip(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(min(max(value, low), high))


def _exp_score(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.exp(-(value * value)))


def score_case_csv(
    csv_path: str | Path,
    *,
    case_id: str | None = None,
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Compute a scalar guiding score from one per-case guiding_metrics.csv.

    The score is designed for capillary guiding scans where the useful target is
    high a0 near the plateau exit, with limited waist growth across the plateau.
    It deliberately ignores late post-capillary propagation unless it falls inside
    the configurable exit-after window.
    """

    cfg = config or ScoreConfig()
    path = Path(csv_path)
    cid = case_id or path.parent.name

    base: dict[str, Any] = {
        "case_id": cid,
        "csv_path": str(path),
        "status": "failed",
        "failure_reason": "",
        "score": float("nan"),
    }

    if not path.is_file():
        return {**base, "failure_reason": "missing_csv"}

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {**base, "failure_reason": f"read_csv_failed: {exc}"}

    missing = [col for col in REQUIRED_SCORE_COLUMNS if col not in df.columns]
    if missing:
        return {**base, "failure_reason": f"missing_columns: {missing}"}

    if df.empty:
        return {**base, "failure_reason": "empty_csv"}

    df = df.copy()
    for col in REQUIRED_SCORE_COLUMNS:
        df[col] = _to_float_series(df, col)

    window = infer_plateau_window_mm_from_text(str(path))
    if window is None:
        return {**base, "failure_reason": "could_not_infer_plateau_window"}

    plateau_start_mm, plateau_end_mm = window
    entry_start_mm = plateau_start_mm
    entry_end_mm = plateau_start_mm + cfg.entry_window_mm
    exit_start_mm = plateau_end_mm - cfg.exit_before_mm
    exit_end_mm = plateau_end_mm + cfg.exit_after_mm
    analysis_start_mm = plateau_start_mm
    analysis_end_mm = exit_end_mm

    z = df["propagation_mm"]
    entry = df[(z >= entry_start_mm) & (z <= entry_end_mm)]
    exit_ = df[(z >= exit_start_mm) & (z <= exit_end_mm)]
    analysis = df[(z >= analysis_start_mm) & (z <= analysis_end_mm)]

    if entry.empty:
        return {**base, "failure_reason": "empty_entry_window"}
    if exit_.empty:
        return {**base, "failure_reason": "empty_exit_window"}
    if analysis.empty:
        return {**base, "failure_reason": "empty_analysis_window"}

    a0_exit = _finite_median(exit_["a0_peak"])
    waist_entry_um = _finite_median(entry["waist_um"])
    waist_exit_um = _finite_median(exit_["waist_um"])

    a0_max, a0_max_local_idx = _finite_max(analysis["a0_peak"])
    if a0_max_local_idx is None:
        a0_max_mm = float("nan")
    else:
        a0_max_mm = float(analysis.iloc[a0_max_local_idx]["propagation_mm"])

    waist_arr = pd.to_numeric(analysis["waist_um"], errors="coerce").to_numpy(float)
    waist_finite = waist_arr[np.isfinite(waist_arr) & (waist_arr > 0.0)]
    if waist_finite.size >= 2:
        waist_jitter_log = float(np.std(np.log(waist_finite)))
    else:
        waist_jitter_log = float("nan")

    valid_mask = np.isfinite(
        pd.to_numeric(analysis["a0_peak"], errors="coerce").to_numpy(float)
    ) & np.isfinite(
        pd.to_numeric(analysis["waist_um"], errors="coerce").to_numpy(float)
    )
    valid_fraction = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
    nan_fraction = 1.0 - valid_fraction

    required_values = [a0_exit, waist_entry_um, waist_exit_um, a0_max, waist_jitter_log]
    if not all(np.isfinite(v) for v in required_values):
        return {
            **base,
            "failure_reason": "non_finite_score_metric",
            "plateau_start_mm": plateau_start_mm,
            "plateau_end_mm": plateau_end_mm,
            "a0_exit": a0_exit,
            "waist_entry_um": waist_entry_um,
            "waist_exit_um": waist_exit_um,
            "a0_max_analysis": a0_max,
            "waist_jitter_log": waist_jitter_log,
            "valid_fraction": valid_fraction,
            "nan_fraction": nan_fraction,
        }

    if waist_entry_um <= 0.0 or a0_max <= 0.0:
        return {
            **base,
            "failure_reason": "non_positive_reference_metric",
            "waist_entry_um": waist_entry_um,
            "a0_max_analysis": a0_max,
        }

    waist_growth = waist_exit_um / waist_entry_um
    a0_exit_over_analysis_max = a0_exit / a0_max

    a0_exit_component = _clip(a0_exit / cfg.a0_target, 0.0, cfg.a0_component_cap)
    a0_retention_component = _clip(a0_exit_over_analysis_max, 0.0, 1.0)
    waist_growth_component = _exp_score(
        max(0.0, waist_growth - 1.0) / cfg.waist_growth_sigma
    )
    waist_stability_component = _exp_score(waist_jitter_log / cfg.waist_jitter_sigma)

    weighted = (
        cfg.weight_a0_exit * a0_exit_component
        + cfg.weight_a0_retention * a0_retention_component
        + cfg.weight_waist_growth * waist_growth_component
        + cfg.weight_waist_stability * waist_stability_component
    )

    score = 100.0 * weighted * valid_fraction

    return {
        **base,
        "status": "ok",
        "failure_reason": "",
        "score": float(score),
        "plateau_start_mm": plateau_start_mm,
        "plateau_end_mm": plateau_end_mm,
        "entry_start_mm": entry_start_mm,
        "entry_end_mm": entry_end_mm,
        "exit_start_mm": exit_start_mm,
        "exit_end_mm": exit_end_mm,
        "analysis_start_mm": analysis_start_mm,
        "analysis_end_mm": analysis_end_mm,
        "n_rows": int(len(df)),
        "n_entry_rows": int(len(entry)),
        "n_exit_rows": int(len(exit_)),
        "n_analysis_rows": int(len(analysis)),
        "valid_fraction": valid_fraction,
        "nan_fraction": nan_fraction,
        "a0_exit": a0_exit,
        "a0_max_analysis": a0_max,
        "a0_max_analysis_mm": a0_max_mm,
        "a0_exit_over_analysis_max": a0_exit_over_analysis_max,
        "waist_entry_um": waist_entry_um,
        "waist_exit_um": waist_exit_um,
        "waist_growth": waist_growth,
        "waist_jitter_log": waist_jitter_log,
        "component_a0_exit": a0_exit_component,
        "component_a0_retention": a0_retention_component,
        "component_waist_growth": waist_growth_component,
        "component_waist_stability": waist_stability_component,
        "weight_a0_exit": cfg.weight_a0_exit,
        "weight_a0_retention": cfg.weight_a0_retention,
        "weight_waist_growth": cfg.weight_waist_growth,
        "weight_waist_stability": cfg.weight_waist_stability,
    }


@dataclass(frozen=True)
class TripletScoreConfig:
    reference_deadband: float = 5.0
    reference_scale: float = 15.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _signed_deadband(delta: float, deadband: float) -> float:
    """Remove a symmetric deadband around zero while preserving sign."""
    if not np.isfinite(delta):
        return float("nan")

    abs_delta = abs(delta)
    if abs_delta <= deadband:
        return 0.0

    return float(np.sign(delta) * (abs_delta - deadband))


def reference_factor_from_delta(
    delta: float,
    *,
    deadband: float = 5.0,
    scale: float = 15.0,
) -> float:
    """Map channel-reference score difference to [-1, 1].

    Positive means the channel beats the best reference.
    Near zero means the channel is indistinguishable from the references.
    Negative means a reference beats the channel.
    """
    if not np.isfinite(delta):
        return float("nan")

    if scale <= 0.0:
        raise ValueError("scale must be positive")

    effective_delta = _signed_deadband(delta, deadband)
    return float(np.tanh(effective_delta / scale))


def score_triplet_csvs(
    *,
    channel_csv: str | Path,
    uniform_csv: str | Path,
    vacuum_csv: str | Path,
    channel_case_id: str | None = None,
    uniform_case_id: str | None = None,
    vacuum_case_id: str | None = None,
    case_config: ScoreConfig | None = None,
    triplet_config: TripletScoreConfig | None = None,
) -> dict[str, Any]:
    """Compute a reference-aware guiding score for one channel/uniform/vacuum triplet.

    The final score is positive only when the channel beats both references.
    If the channel is similar to the best reference, the final score is near zero.
    If either uniform or vacuum beats the channel, the final score is negative.
    """

    tcfg = triplet_config or TripletScoreConfig()

    channel = score_case_csv(
        channel_csv,
        case_id=channel_case_id,
        config=case_config,
    )
    uniform = score_case_csv(
        uniform_csv,
        case_id=uniform_case_id,
        config=case_config,
    )
    vacuum = score_case_csv(
        vacuum_csv,
        case_id=vacuum_case_id,
        config=case_config,
    )

    base: dict[str, Any] = {
        "status": "failed",
        "failure_reason": "",
        "channel_case_id": channel.get("case_id"),
        "uniform_case_id": uniform.get("case_id"),
        "vacuum_case_id": vacuum.get("case_id"),
        "score_channel": channel.get("score", float("nan")),
        "score_uniform": uniform.get("score", float("nan")),
        "score_vacuum": vacuum.get("score", float("nan")),
        "final_score": float("nan"),
    }

    failed = []
    for name, result in [
        ("channel", channel),
        ("uniform", uniform),
        ("vacuum", vacuum),
    ]:
        if result.get("status") != "ok":
            failed.append(f"{name}: {result.get('failure_reason', 'unknown')}")

    if failed:
        return {
            **base,
            "failure_reason": "; ".join(failed),
        }

    score_channel = float(channel["score"])
    score_uniform = float(uniform["score"])
    score_vacuum = float(vacuum["score"])

    if score_uniform >= score_vacuum:
        reference_kind = "uniform"
        reference_score = score_uniform
    else:
        reference_kind = "vacuum"
        reference_score = score_vacuum

    score_delta_vs_reference = score_channel - reference_score
    score_delta_vs_uniform = score_channel - score_uniform
    score_delta_vs_vacuum = score_channel - score_vacuum

    reference_factor = reference_factor_from_delta(
        score_delta_vs_reference,
        deadband=tcfg.reference_deadband,
        scale=tcfg.reference_scale,
    )

    final_score = score_channel * reference_factor

    return {
        **base,
        "status": "ok",
        "failure_reason": "",
        "reference_kind": reference_kind,
        "reference_score": reference_score,
        "score_delta_vs_reference": score_delta_vs_reference,
        "score_delta_vs_uniform": score_delta_vs_uniform,
        "score_delta_vs_vacuum": score_delta_vs_vacuum,
        "reference_factor": reference_factor,
        "reference_deadband": tcfg.reference_deadband,
        "reference_scale": tcfg.reference_scale,
        "final_score": final_score,
        # Useful physical diagnostics for inspection.
        "a0_exit_channel": channel.get("a0_exit", float("nan")),
        "a0_exit_uniform": uniform.get("a0_exit", float("nan")),
        "a0_exit_vacuum": vacuum.get("a0_exit", float("nan")),
        "a0_exit_channel_over_uniform": _safe_ratio(
            float(channel.get("a0_exit", float("nan"))),
            float(uniform.get("a0_exit", float("nan"))),
        ),
        "a0_exit_channel_over_vacuum": _safe_ratio(
            float(channel.get("a0_exit", float("nan"))),
            float(vacuum.get("a0_exit", float("nan"))),
        ),
        "waist_growth_channel": channel.get("waist_growth", float("nan")),
        "waist_growth_uniform": uniform.get("waist_growth", float("nan")),
        "waist_growth_vacuum": vacuum.get("waist_growth", float("nan")),
        "waist_growth_channel_over_uniform": _safe_ratio(
            float(channel.get("waist_growth", float("nan"))),
            float(uniform.get("waist_growth", float("nan"))),
        ),
        "waist_growth_channel_over_vacuum": _safe_ratio(
            float(channel.get("waist_growth", float("nan"))),
            float(vacuum.get("waist_growth", float("nan"))),
        ),
        "a0_retention_channel": channel.get("a0_exit_over_analysis_max", float("nan")),
        "a0_retention_uniform": uniform.get("a0_exit_over_analysis_max", float("nan")),
        "a0_retention_vacuum": vacuum.get("a0_exit_over_analysis_max", float("nan")),
        "valid_fraction_channel": channel.get("valid_fraction", float("nan")),
        "valid_fraction_uniform": uniform.get("valid_fraction", float("nan")),
        "valid_fraction_vacuum": vacuum.get("valid_fraction", float("nan")),
    }
