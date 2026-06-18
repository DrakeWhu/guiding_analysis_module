from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .beamlike import add_beamlike_metrics


@dataclass(frozen=True)
class BeamlikePairConfig:
    """Configuration for channel-vs-uniform beam comparison.

    The absolute beamlike_score remains a per-case metric. This comparison layer
    asks whether a channel case beats its matched uniform reference.
    """

    reference_deadband_log: float = math.log(1.05)
    reference_scale_log: float = math.log(1.5)
    score_floor: float = 1.0


PAIR_OUTPUT_COLUMNS = [
    "status",
    "failure_reason",
    "comparison_status",
    "channel_case_id",
    "uniform_case_id",
    "channel_csv",
    "uniform_csv",
    "row_selection",
    "beamlike_score_source_channel",
    "beamlike_score_source_uniform",
    "channel_iteration",
    "uniform_iteration",
    "eligible_beamlike_channel",
    "eligible_beamlike_uniform",
    "beamlike_status_channel",
    "beamlike_status_uniform",
    "beamlike_rejection_reasons_channel",
    "beamlike_rejection_reasons_uniform",
    "beamlike_score_channel",
    "beamlike_score_uniform",
    "beamlike_score_delta",
    "beamlike_score_ratio",
    "beamlike_score_log_advantage",
    "beamlike_reference_factor",
    "beamlike_reference_scale_score",
    "beamlike_gain_score",
    "beam_yield_score_channel",
    "beam_yield_score_uniform",
    "beam_yield_score_delta",
    "beam_yield_score_ratio",
    "charge_hot_pC_channel",
    "charge_hot_pC_uniform",
    "charge_hot_pC_delta",
    "charge_hot_pC_ratio",
    "n_macroparticles_hot_channel",
    "n_macroparticles_hot_uniform",
    "n_macroparticles_hot_delta",
    "n_macroparticles_hot_ratio",
    "E95_hot_MeV_channel",
    "E95_hot_MeV_uniform",
    "E95_hot_MeV_delta",
    "E95_hot_MeV_ratio",
    "Emean_hot_MeV_channel",
    "Emean_hot_MeV_uniform",
    "Emean_hot_MeV_delta",
    "Emean_hot_MeV_ratio",
    "Emax_hot_MeV_channel",
    "Emax_hot_MeV_uniform",
    "Emax_hot_MeV_delta",
    "Emax_hot_MeV_ratio",
    "mono_proxy_E95_over_Emax_channel",
    "mono_proxy_E95_over_Emax_uniform",
    "mono_proxy_E95_over_Emax_delta",
    "mono_proxy_E95_over_Emax_ratio",
    "z_span_hot_mm_channel",
    "z_span_hot_mm_uniform",
    "z_span_hot_mm_delta",
    "z_span_hot_mm_ratio",
    "theta_rms_mrad_channel",
    "theta_rms_mrad_uniform",
    "theta_rms_mrad_delta",
    "theta_rms_mrad_ratio",
    "divergence_improvement_mrad",
]


def _finite_float(
    row: dict[str, Any], key: str, default: float = float("nan")
) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _string_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)


def _boolish(row: dict[str, Any], key: str) -> bool | str:
    value = row.get(key, "")
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return ""


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _signed_deadband(value: float, deadband: float) -> float:
    if not math.isfinite(value):
        return float("nan")

    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0

    return math.copysign(magnitude - deadband, value)


def _reference_factor_from_log_advantage(
    log_advantage: float,
    *,
    deadband_log: float,
    scale_log: float,
) -> float:
    if not math.isfinite(log_advantage):
        return float("nan")
    if scale_log <= 0.0:
        raise ValueError("scale_log must be positive")

    return math.tanh(_signed_deadband(log_advantage, deadband_log) / scale_log)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f_in:
        return list(csv.DictReader(f_in))


def _ensure_beamlike_metrics(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return a row with beamlike metrics.

    Existing new-format particle_summary.csv files already contain beamlike_score.
    Older summaries from campaigns analyzed before the beamlike integration only
    contain raw particle metrics. In that case, compute the beamlike columns
    on-the-fly without rewriting the original CSV.
    """
    value = row.get("beamlike_score", "")
    if str(value).strip() != "":
        return dict(row), "particle_summary"

    return add_beamlike_metrics(row), "computed_on_the_fly"


def _select_summary_row(
    rows: list[dict[str, str]],
    *,
    row_selection: str,
    csv_path: str | Path,
) -> dict[str, str]:
    if not rows:
        raise ValueError(f"empty_csv: {csv_path}")

    if row_selection == "single":
        if len(rows) != 1:
            raise ValueError(
                f"expected_single_row_but_found_{len(rows)}: {csv_path}. "
                "Use --row-selection last or --row-selection max-beamlike explicitly."
            )
        return rows[0]

    if row_selection == "last":
        return sorted(rows, key=lambda r: _finite_float(r, "iteration", -math.inf))[-1]

    if row_selection == "max-beamlike":
        return sorted(
            rows, key=lambda r: _finite_float(r, "beamlike_score", -math.inf)
        )[-1]

    raise ValueError(f"Unknown row_selection: {row_selection}")


def read_particle_summary_row(
    csv_path: str | Path,
    *,
    row_selection: str = "single",
) -> dict[str, str]:
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"missing_csv: {path}")

    return _select_summary_row(
        _read_csv_rows(path),
        row_selection=row_selection,
        csv_path=path,
    )


def _metric_pair(
    out: dict[str, Any],
    *,
    channel: dict[str, Any],
    uniform: dict[str, Any],
    metric: str,
) -> None:
    ch = _finite_float(channel, metric)
    uni = _finite_float(uniform, metric)

    out[f"{metric}_channel"] = ch
    out[f"{metric}_uniform"] = uni
    out[f"{metric}_delta"] = (
        ch - uni if math.isfinite(ch) and math.isfinite(uni) else float("nan")
    )
    out[f"{metric}_ratio"] = _safe_ratio(ch, uni)


def compare_beamlike_pair_rows(
    *,
    channel: dict[str, Any],
    uniform: dict[str, Any],
    channel_case_id: str | None = None,
    uniform_case_id: str | None = None,
    channel_csv: str | Path | None = None,
    uniform_csv: str | Path | None = None,
    row_selection: str = "single",
    config: BeamlikePairConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BeamlikePairConfig()

    score_ch = _finite_float(channel, "beamlike_score", 0.0)
    score_uni = _finite_float(uniform, "beamlike_score", 0.0)

    # log comparison with an explicit floor avoids infinities when the uniform
    # reference has zero score, while still rewarding a real channel-only beam.
    log_advantage = math.log(
        (score_ch + cfg.score_floor) / (score_uni + cfg.score_floor)
    )

    reference_factor = _reference_factor_from_log_advantage(
        log_advantage,
        deadband_log=cfg.reference_deadband_log,
        scale_log=cfg.reference_scale_log,
    )

    reference_scale_score = max(score_ch, score_uni)

    gain_score = (
        reference_scale_score * reference_factor
        if math.isfinite(reference_factor)
        else float("nan")
    )

    out: dict[str, Any] = {
        "status": "ok",
        "failure_reason": "",
        "comparison_status": "ok",
        "channel_case_id": channel_case_id or _string_value(channel, "case_id"),
        "uniform_case_id": uniform_case_id or _string_value(uniform, "case_id"),
        "channel_csv": "" if channel_csv is None else str(channel_csv),
        "uniform_csv": "" if uniform_csv is None else str(uniform_csv),
        "row_selection": row_selection,
        "channel_iteration": _finite_float(channel, "iteration"),
        "uniform_iteration": _finite_float(uniform, "iteration"),
        "eligible_beamlike_channel": _boolish(channel, "eligible_beamlike"),
        "eligible_beamlike_uniform": _boolish(uniform, "eligible_beamlike"),
        "beamlike_status_channel": _string_value(channel, "beamlike_status"),
        "beamlike_status_uniform": _string_value(uniform, "beamlike_status"),
        "beamlike_rejection_reasons_channel": _string_value(
            channel,
            "beamlike_rejection_reasons",
        ),
        "beamlike_rejection_reasons_uniform": _string_value(
            uniform,
            "beamlike_rejection_reasons",
        ),
        "beamlike_score_channel": score_ch,
        "beamlike_score_uniform": score_uni,
        "beamlike_score_delta": score_ch - score_uni,
        "beamlike_score_ratio": _safe_ratio(score_ch, score_uni),
        "beamlike_score_log_advantage": log_advantage,
        "beamlike_reference_factor": reference_factor,
        "beamlike_gain_score": gain_score,
        "beamlike_reference_scale_score": reference_scale_score,
    }

    for metric in [
        "beam_yield_score",
        "charge_hot_pC",
        "n_macroparticles_hot",
        "E95_hot_MeV",
        "Emean_hot_MeV",
        "Emax_hot_MeV",
        "mono_proxy_E95_over_Emax",
        "z_span_hot_mm",
        "theta_rms_mrad",
    ]:
        _metric_pair(out, channel=channel, uniform=uniform, metric=metric)

    theta_ch = out["theta_rms_mrad_channel"]
    theta_uni = out["theta_rms_mrad_uniform"]
    out["divergence_improvement_mrad"] = (
        theta_uni - theta_ch
        if math.isfinite(theta_ch) and math.isfinite(theta_uni)
        else float("nan")
    )

    return out


def compare_beamlike_pair_csvs(
    *,
    channel_csv: str | Path,
    uniform_csv: str | Path,
    channel_case_id: str | None = None,
    uniform_case_id: str | None = None,
    row_selection: str = "single",
    config: BeamlikePairConfig | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "failed",
        "failure_reason": "",
        "comparison_status": "failed",
        "channel_case_id": channel_case_id or "",
        "uniform_case_id": uniform_case_id or "",
        "channel_csv": str(channel_csv),
        "uniform_csv": str(uniform_csv),
        "row_selection": row_selection,
        "beamlike_score_source_channel": "",
        "beamlike_score_source_uniform": "",
        "beamlike_gain_score": float("nan"),
    }

    try:
        channel = read_particle_summary_row(channel_csv, row_selection=row_selection)
        uniform = read_particle_summary_row(uniform_csv, row_selection=row_selection)
    except Exception as exc:
        return {**base, "failure_reason": str(exc)}

    required_raw = [
        "charge_hot_pC",
        "n_macroparticles_hot",
        "E95_hot_MeV",
        "Emean_hot_MeV",
        "Emax_hot_MeV",
    ]

    missing = []
    for label, row in [("channel", channel), ("uniform", uniform)]:
        for col in required_raw:
            if col not in row:
                missing.append(f"{label}:{col}")

    if missing:
        return {**base, "failure_reason": f"missing_columns: {missing}"}

    channel, channel_source = _ensure_beamlike_metrics(channel)
    uniform, uniform_source = _ensure_beamlike_metrics(uniform)

    row = compare_beamlike_pair_rows(
        channel=channel,
        uniform=uniform,
        channel_case_id=channel_case_id,
        uniform_case_id=uniform_case_id,
        channel_csv=channel_csv,
        uniform_csv=uniform_csv,
        row_selection=row_selection,
        config=config,
    )

    row["beamlike_score_source_channel"] = channel_source
    row["beamlike_score_source_uniform"] = uniform_source

    return row


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)

    fieldnames: list[str] = []

    for preferred in PAIR_OUTPUT_COLUMNS:
        for row in rows:
            if preferred in row and preferred not in fieldnames:
                fieldnames.append(preferred)
                break

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
