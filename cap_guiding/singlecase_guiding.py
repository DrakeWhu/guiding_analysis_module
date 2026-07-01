from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .case_metadata import infer_plateau_window_mm_from_text


SINGLECASE_GUIDING_SCORE_FILENAME = "guiding_singlecase_score.csv"
SINGLECASE_GUIDING_SCHEMA_VERSION = "guiding_singlecase_score_v1"
SINGLECASE_GUIDING_CONFIG_ID = "guiding_singlecase_score_v1_waist_retention_20260701"

REQUIRED_SINGLECASE_GUIDING_COLUMNS = [
    "iteration",
    "propagation_mm",
    "a0_peak",
    "waist_um",
]


@dataclass(frozen=True)
class SingleCaseGuidingMetricConfig:
    """Configuration for the single-case guiding metric.

    The metric uses only reduced guiding_metrics.csv columns. It is intended to
    score whether one simulation keeps the laser guided across the plateau,
    without comparing against uniform/vacuum references and without reading raw
    WarpX/openPMD diagnostics.

    Waist growth and a0 retention carry the dominant weight. Waist oscillations
    are kept as a weak diagnostic penalty because imperfect matching can produce
    oscillations that may still be physically interesting.
    """

    schema_version: str = SINGLECASE_GUIDING_SCHEMA_VERSION
    config_id: str = SINGLECASE_GUIDING_CONFIG_ID

    plateau_start_default_mm: float = 5.0
    entry_window_mm: float = 1.0
    exit_window_mm: float = 1.0
    min_valid_plateau_rows: int = 3

    # a0 local drops of order 30% start to matter clearly.
    sigma_a0_drop_log: float = float(np.log(1.30))

    # Small waist growth is tolerated. Growth beyond this deadband is penalized.
    waist_growth_deadband: float = 1.10
    sigma_waist_growth_log: float = float(np.log(1.50))

    # Waist oscillations are weakly weighted and moderately tolerated.
    sigma_waist_jitter_log: float = float(np.log(1.35))

    weight_a0_retention: float = 0.35
    weight_a0_stability: float = 0.15
    weight_waist_growth: float = 0.45
    weight_waist_stability: float = 0.05


def _base_result(
    *,
    case_id: str | None,
    csv_path: str | Path | None,
    config: SingleCaseGuidingMetricConfig,
) -> dict[str, Any]:
    return {
        "case_id": case_id or "",
        "csv_path": "" if csv_path is None else str(csv_path),
        "metric_guiding_singlecase_schema_version": config.schema_version,
        "metric_guiding_singlecase_config_id": config.config_id,
        "metric_guiding_singlecase_status": "failed",
        "metric_guiding_singlecase_failure_reason": "",
        "metric_guiding_singlecase_score_v1": float("nan"),
    }


def _as_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _finite_median(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(min(max(value, 0.0), 1.0))


def _exp_quadratic_component(value: float, sigma: float) -> float:
    if not np.isfinite(value) or not np.isfinite(sigma) or sigma <= 0.0:
        return float("nan")
    return float(np.exp(-((value / sigma) ** 2)))


def _positive_log_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    return float(np.log(numerator / denominator))


def _resolve_plateau_window_mm(
    *,
    csv_path: str | Path | None,
    plateau_start_mm: float | None,
    plateau_end_mm: float | None,
    plateau_length_mm: float | None,
    config: SingleCaseGuidingMetricConfig,
) -> tuple[float, float, str] | None:
    if plateau_start_mm is not None and plateau_end_mm is not None:
        return float(plateau_start_mm), float(plateau_end_mm), "explicit_start_end"

    if plateau_length_mm is not None:
        start = (
            float(plateau_start_mm)
            if plateau_start_mm is not None
            else float(config.plateau_start_default_mm)
        )
        return start, start + float(plateau_length_mm), "explicit_length"

    if csv_path is not None:
        inferred = infer_plateau_window_mm_from_text(str(csv_path))
        if inferred is not None:
            start, end = inferred
            return float(start), float(end), "path_inference"

    return None


def _window_or_edge_rows(
    valid_plateau: pd.DataFrame,
    *,
    z_col: str,
    start_mm: float,
    end_mm: float,
    edge: str,
) -> tuple[pd.DataFrame, str]:
    z = valid_plateau[z_col]
    rows = valid_plateau[(z >= start_mm) & (z <= end_mm)]
    if len(rows) > 0:
        return rows, "window"

    if edge == "entry":
        return valid_plateau.head(1), "first_valid_fallback"
    if edge == "exit":
        return valid_plateau.tail(1), "last_valid_fallback"

    raise ValueError(f"Unknown edge: {edge}")


def score_singlecase_guiding_dataframe(
    df: pd.DataFrame,
    *,
    case_id: str | None = None,
    csv_path: str | Path | None = None,
    plateau_start_mm: float | None = None,
    plateau_end_mm: float | None = None,
    plateau_length_mm: float | None = None,
    config: SingleCaseGuidingMetricConfig | None = None,
) -> dict[str, Any]:
    """Compute guiding_singlecase_score_v1 from one guiding_metrics table.

    This function does not read raw WarpX/openPMD diagnostics. It only consumes
    the reduced per-dump columns already present in guiding_metrics.csv.
    """

    cfg = config or SingleCaseGuidingMetricConfig()
    base = _base_result(case_id=case_id, csv_path=csv_path, config=cfg)

    missing = [col for col in REQUIRED_SINGLECASE_GUIDING_COLUMNS if col not in df]
    if missing:
        return {**base, "metric_guiding_singlecase_failure_reason": f"missing_columns: {missing}"}

    if df.empty:
        return {**base, "metric_guiding_singlecase_failure_reason": "empty_csv"}

    window = _resolve_plateau_window_mm(
        csv_path=csv_path,
        plateau_start_mm=plateau_start_mm,
        plateau_end_mm=plateau_end_mm,
        plateau_length_mm=plateau_length_mm,
        config=cfg,
    )
    if window is None:
        return {
            **base,
            "metric_guiding_singlecase_failure_reason": "could_not_determine_plateau_window",
        }

    plateau_start, plateau_end, plateau_policy = window
    if not np.isfinite(plateau_start) or not np.isfinite(plateau_end):
        return {
            **base,
            "metric_guiding_singlecase_failure_reason": "non_finite_plateau_window",
            "metric_guiding_plateau_start_mm": plateau_start,
            "metric_guiding_plateau_end_mm": plateau_end,
            "metric_guiding_plateau_policy": plateau_policy,
        }
    if plateau_end <= plateau_start:
        return {
            **base,
            "metric_guiding_singlecase_failure_reason": "invalid_plateau_window",
            "metric_guiding_plateau_start_mm": plateau_start,
            "metric_guiding_plateau_end_mm": plateau_end,
            "metric_guiding_plateau_policy": plateau_policy,
        }

    work = df.copy()
    for col in REQUIRED_SINGLECASE_GUIDING_COLUMNS:
        work[col] = _as_numeric(work, col)

    work = work.sort_values("propagation_mm").reset_index(drop=True)
    z = work["propagation_mm"]
    plateau = work[(z >= plateau_start) & (z <= plateau_end)].copy()

    n_rows = int(len(work))
    n_plateau_rows = int(len(plateau))

    common = {
        "metric_guiding_plateau_start_mm": float(plateau_start),
        "metric_guiding_plateau_end_mm": float(plateau_end),
        "metric_guiding_plateau_policy": plateau_policy,
        "metric_guiding_n_rows": n_rows,
        "metric_guiding_n_plateau_rows": n_plateau_rows,
    }

    if plateau.empty:
        return {
            **base,
            **common,
            "metric_guiding_singlecase_failure_reason": "empty_plateau_window",
        }

    valid_mask = (
        np.isfinite(plateau["propagation_mm"].to_numpy(float))
        & np.isfinite(plateau["a0_peak"].to_numpy(float))
        & np.isfinite(plateau["waist_um"].to_numpy(float))
        & (plateau["a0_peak"].to_numpy(float) > 0.0)
        & (plateau["waist_um"].to_numpy(float) > 0.0)
    )
    valid_plateau = plateau[valid_mask].copy().sort_values("propagation_mm")
    n_valid = int(len(valid_plateau))
    valid_fraction = float(n_valid / n_plateau_rows) if n_plateau_rows else 0.0

    common = {
        **common,
        "metric_guiding_n_valid_plateau_rows": n_valid,
        "metric_guiding_valid_fraction": valid_fraction,
    }

    if n_valid < cfg.min_valid_plateau_rows:
        return {
            **base,
            **common,
            "metric_guiding_singlecase_failure_reason": "not_enough_valid_plateau_rows",
        }

    z_valid = valid_plateau["propagation_mm"].to_numpy(float)
    span_fraction = _clip01((float(np.max(z_valid)) - float(np.min(z_valid))) / (plateau_end - plateau_start))
    coverage_component = float(span_fraction * valid_fraction)

    entry_end = plateau_start + cfg.entry_window_mm
    exit_start = plateau_end - cfg.exit_window_mm

    entry_rows, entry_policy = _window_or_edge_rows(
        valid_plateau,
        z_col="propagation_mm",
        start_mm=plateau_start,
        end_mm=entry_end,
        edge="entry",
    )
    exit_rows, exit_policy = _window_or_edge_rows(
        valid_plateau,
        z_col="propagation_mm",
        start_mm=exit_start,
        end_mm=plateau_end,
        edge="exit",
    )

    a0_entry = _finite_median(entry_rows["a0_peak"])
    a0_exit = _finite_median(exit_rows["a0_peak"])
    a0_values = valid_plateau["a0_peak"].to_numpy(float)
    a0_max = float(np.max(a0_values))

    waist_entry = _finite_median(entry_rows["waist_um"])
    waist_exit = _finite_median(exit_rows["waist_um"])
    waist_values = valid_plateau["waist_um"].to_numpy(float)
    waist_max = float(np.max(waist_values))

    required_positive = [a0_entry, a0_exit, a0_max, waist_entry, waist_exit, waist_max]
    if not all(np.isfinite(v) and v > 0.0 for v in required_positive):
        return {
            **base,
            **common,
            "metric_guiding_singlecase_failure_reason": "non_positive_reference_metric",
            "metric_guiding_a0_entry": a0_entry,
            "metric_guiding_a0_exit": a0_exit,
            "metric_guiding_a0_max": a0_max,
            "metric_guiding_waist_entry_um": waist_entry,
            "metric_guiding_waist_exit_um": waist_exit,
            "metric_guiding_waist_max_um": waist_max,
        }

    a0_retention_ratio = float(a0_exit / a0_entry)
    a0_retention_component = _clip01(a0_retention_ratio)

    if len(a0_values) >= 2:
        log_step_ratios = np.log(a0_values[:-1] / a0_values[1:])
        local_drops = np.maximum(0.0, log_step_ratios)
        a0_drop_rms_log = float(np.sqrt(np.mean(local_drops * local_drops)))
    else:
        a0_drop_rms_log = float("nan")
    a0_stability_component = _exp_quadratic_component(
        a0_drop_rms_log,
        cfg.sigma_a0_drop_log,
    )

    waist_growth_factor = float(waist_max / waist_entry)
    waist_growth_excess_log = max(
        0.0,
        _positive_log_ratio(waist_growth_factor, cfg.waist_growth_deadband),
    )
    waist_growth_component = _exp_quadratic_component(
        waist_growth_excess_log,
        cfg.sigma_waist_growth_log,
    )

    waist_jitter_log = float(np.std(np.log(waist_values)))
    waist_stability_component = _exp_quadratic_component(
        waist_jitter_log,
        cfg.sigma_waist_jitter_log,
    )

    components = [
        a0_retention_component,
        a0_stability_component,
        waist_growth_component,
        waist_stability_component,
        coverage_component,
    ]
    if not all(np.isfinite(v) for v in components):
        return {
            **base,
            **common,
            "metric_guiding_singlecase_failure_reason": "non_finite_component",
            "metric_guiding_a0_retention_component_v1": a0_retention_component,
            "metric_guiding_a0_stability_component_v1": a0_stability_component,
            "metric_guiding_waist_growth_component_v1": waist_growth_component,
            "metric_guiding_waist_stability_component_v1": waist_stability_component,
            "metric_guiding_plateau_coverage_component_v1": coverage_component,
        }

    weights = np.asarray(
        [
            cfg.weight_a0_retention,
            cfg.weight_a0_stability,
            cfg.weight_waist_growth,
            cfg.weight_waist_stability,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        return {
            **base,
            **common,
            "metric_guiding_singlecase_failure_reason": "invalid_component_weights",
        }
    weights = weights / float(np.sum(weights))

    weighted_shape_score = float(
        weights[0] * a0_retention_component
        + weights[1] * a0_stability_component
        + weights[2] * waist_growth_component
        + weights[3] * waist_stability_component
    )
    score = float(100.0 * coverage_component * weighted_shape_score)

    return {
        **base,
        **common,
        "metric_guiding_singlecase_status": "ok",
        "metric_guiding_singlecase_failure_reason": "",
        "metric_guiding_singlecase_score_v1": score,
        "metric_guiding_a0_retention_component_v1": a0_retention_component,
        "metric_guiding_a0_stability_component_v1": a0_stability_component,
        "metric_guiding_waist_growth_component_v1": waist_growth_component,
        "metric_guiding_waist_stability_component_v1": waist_stability_component,
        "metric_guiding_plateau_coverage_component_v1": coverage_component,
        "metric_guiding_a0_entry": a0_entry,
        "metric_guiding_a0_exit": a0_exit,
        "metric_guiding_a0_max": a0_max,
        "metric_guiding_a0_retention_ratio": a0_retention_ratio,
        "metric_guiding_a0_drop_rms_log": a0_drop_rms_log,
        "metric_guiding_waist_entry_um": waist_entry,
        "metric_guiding_waist_exit_um": waist_exit,
        "metric_guiding_waist_max_um": waist_max,
        "metric_guiding_waist_growth_factor": waist_growth_factor,
        "metric_guiding_waist_growth_excess_log": waist_growth_excess_log,
        "metric_guiding_waist_jitter_log": waist_jitter_log,
        "metric_guiding_plateau_span_covered": span_fraction,
        "metric_guiding_entry_policy": entry_policy,
        "metric_guiding_exit_policy": exit_policy,
        "metric_guiding_entry_window_mm": float(cfg.entry_window_mm),
        "metric_guiding_exit_window_mm": float(cfg.exit_window_mm),
        "metric_guiding_min_valid_plateau_rows": int(cfg.min_valid_plateau_rows),
        "metric_guiding_sigma_a0_drop_log": float(cfg.sigma_a0_drop_log),
        "metric_guiding_waist_growth_deadband": float(cfg.waist_growth_deadband),
        "metric_guiding_sigma_waist_growth_log": float(cfg.sigma_waist_growth_log),
        "metric_guiding_sigma_waist_jitter_log": float(cfg.sigma_waist_jitter_log),
        "metric_guiding_weight_a0_retention": float(weights[0]),
        "metric_guiding_weight_a0_stability": float(weights[1]),
        "metric_guiding_weight_waist_growth": float(weights[2]),
        "metric_guiding_weight_waist_stability": float(weights[3]),
    }


def score_singlecase_guiding_csv(
    csv_path: str | Path,
    *,
    case_id: str | None = None,
    plateau_start_mm: float | None = None,
    plateau_end_mm: float | None = None,
    plateau_length_mm: float | None = None,
    config: SingleCaseGuidingMetricConfig | None = None,
) -> dict[str, Any]:
    """Read one guiding_metrics.csv and compute guiding_singlecase_score_v1."""

    cfg = config or SingleCaseGuidingMetricConfig()
    path = Path(csv_path)
    cid = case_id or path.parent.name
    base = _base_result(case_id=cid, csv_path=path, config=cfg)

    if not path.is_file():
        return {**base, "metric_guiding_singlecase_failure_reason": "missing_csv"}

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            **base,
            "metric_guiding_singlecase_failure_reason": f"read_csv_failed: {exc}",
        }

    return score_singlecase_guiding_dataframe(
        df,
        case_id=cid,
        csv_path=path,
        plateau_start_mm=plateau_start_mm,
        plateau_end_mm=plateau_end_mm,
        plateau_length_mm=plateau_length_mm,
        config=cfg,
    )


def write_singlecase_guiding_score_csv(
    result: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Write a one-row guiding_singlecase_score.csv sidecar."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False)
    return path


def ensure_singlecase_guiding_score_csv(
    guiding_metrics_csv: str | Path,
    *,
    out_path: str | Path | None = None,
    case_id: str | None = None,
    plateau_start_mm: float | None = None,
    plateau_end_mm: float | None = None,
    plateau_length_mm: float | None = None,
    config: SingleCaseGuidingMetricConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Ensure the sidecar score exists for one reduced guiding_metrics.csv.

    This reads only guiding_metrics.csv. It never reads WarpX/openPMD HDF5 files.
    """

    csv_path = Path(guiding_metrics_csv)
    score_path = (
        Path(out_path)
        if out_path is not None
        else csv_path.parent / SINGLECASE_GUIDING_SCORE_FILENAME
    )

    if score_path.exists() and not overwrite:
        print(f"[USE] existing single-case guiding score: {score_path}")
        return score_path

    result = score_singlecase_guiding_csv(
        csv_path,
        case_id=case_id,
        plateau_start_mm=plateau_start_mm,
        plateau_end_mm=plateau_end_mm,
        plateau_length_mm=plateau_length_mm,
        config=config,
    )
    write_singlecase_guiding_score_csv(result, score_path)
    print(f"[OK] wrote {score_path}")
    return score_path
