from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAIR_KEYS = ["channel_case_id", "uniform_case_id"]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _prefix_non_key_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    keys: list[str],
) -> pd.DataFrame:
    renamed = {col: f"{prefix}{col}" for col in df.columns if col not in keys}
    return df.rename(columns=renamed)


def _bucket_from_factor(status: Any, factor: Any) -> str:
    status_text = str(status).strip().lower()
    if status_text != "ok":
        return "failed"

    x = _finite_float(factor)
    if not math.isfinite(x):
        return "failed"
    if x > 0.0:
        return "positive"
    if x < 0.0:
        return "negative"
    return "neutral"


def _beam_bucket(row: pd.Series) -> str:
    status = row.get("beam_status", "")

    existing = str(row.get("beam_comparison_bucket", "")).strip().lower()
    if existing in {"positive", "neutral", "negative", "failed"}:
        if str(status).strip().lower() == "ok" or existing == "failed":
            return existing

    return _bucket_from_factor(status, row.get("beam_beamlike_reference_factor"))


def _joint_bucket(guiding_bucket: str, beam_bucket: str) -> str:
    if guiding_bucket == "failed" or beam_bucket == "failed":
        return "failed"
    return f"guiding_{guiding_bucket}__beam_{beam_bucket}"


def _nonnegative_finite(value: Any) -> float:
    x = _finite_float(value, 0.0)
    if not math.isfinite(x):
        return 0.0
    return max(x, 0.0)


def load_and_join_guiding_beamlike(
    *,
    triplet_scores_csv: str | Path,
    beamlike_pair_scores_csv: str | Path,
    join_how: str = "inner",
) -> pd.DataFrame:
    """Join guiding triplet scores with beamlike channel-vs-uniform scores.

    The join key is the physical channel/uniform pair. The guiding score may also
    contain a vacuum reference, but the beamlike layer currently compares channel
    vs uniform only.
    """
    triplet_path = Path(triplet_scores_csv)
    beam_path = Path(beamlike_pair_scores_csv)

    guiding = pd.read_csv(triplet_path)
    beam = pd.read_csv(beam_path)

    for label, df in [("guiding", guiding), ("beamlike", beam)]:
        missing = [key for key in PAIR_KEYS if key not in df.columns]
        if missing:
            raise ValueError(f"{label} CSV missing join keys: {missing}")

    guiding = _prefix_non_key_columns(guiding, prefix="guiding_", keys=PAIR_KEYS)
    beam = _prefix_non_key_columns(beam, prefix="beam_", keys=PAIR_KEYS)

    joined = pd.merge(
        guiding,
        beam,
        on=PAIR_KEYS,
        how=join_how,
        validate="many_to_many",
    )

    if joined.empty:
        return joined

    guiding_buckets = []
    beam_buckets = []
    joint_buckets = []
    joint_statuses = []
    alignment_factors = []
    joint_positive_scores = []

    for _, row in joined.iterrows():
        guiding_bucket = _bucket_from_factor(
            row.get("guiding_status", ""),
            row.get("guiding_reference_factor", float("nan")),
        )
        beam_bucket = _beam_bucket(row)
        joint_bucket = _joint_bucket(guiding_bucket, beam_bucket)

        guiding_factor = _finite_float(row.get("guiding_reference_factor"))
        beam_factor = _finite_float(row.get("beam_beamlike_reference_factor"))

        if math.isfinite(guiding_factor) and math.isfinite(beam_factor):
            alignment = guiding_factor * beam_factor
        else:
            alignment = float("nan")

        guiding_final = _nonnegative_finite(row.get("guiding_final_score"))
        beam_gain = _nonnegative_finite(row.get("beam_beamlike_gain_score"))

        # Only rewards cases where both guiding and beam comparison are positive.
        joint_positive = math.sqrt(guiding_final * beam_gain)

        guiding_buckets.append(guiding_bucket)
        beam_buckets.append(beam_bucket)
        joint_buckets.append(joint_bucket)
        joint_statuses.append("ok" if joint_bucket != "failed" else "failed")
        alignment_factors.append(alignment)
        joint_positive_scores.append(joint_positive)

    joined["guiding_bucket"] = guiding_buckets
    joined["beam_bucket"] = beam_buckets
    joined["joint_bucket"] = joint_buckets
    joined["joint_status"] = joint_statuses
    joined["guiding_beam_alignment_factor"] = alignment_factors
    joined["joint_positive_score"] = joint_positive_scores

    return joined


def compute_joint_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple Pearson/Spearman correlations for useful metric pairs."""
    pairs = [
        (
            "guiding_final_score",
            "beam_beamlike_gain_score",
            "guiding final score vs beam gain score",
        ),
        (
            "guiding_reference_factor",
            "beam_beamlike_reference_factor",
            "guiding reference factor vs beam reference factor",
        ),
        (
            "guiding_score_channel",
            "beam_beamlike_score_channel",
            "absolute guiding score vs absolute beamlike score",
        ),
        (
            "guiding_a0_exit_channel",
            "beam_E95_hot_MeV_channel",
            "a0 exit vs hot E95",
        ),
        (
            "guiding_a0_exit_channel",
            "beam_charge_hot_pC_channel",
            "a0 exit vs hot charge",
        ),
        (
            "guiding_waist_growth_channel",
            "beam_z_span_hot_mm_channel",
            "waist growth vs hot z span",
        ),
        (
            "guiding_fraction_a0_beats_reference",
            "beam_beamlike_reference_factor",
            "fraction a0 beats reference vs beam reference factor",
        ),
    ]

    rows: list[dict[str, Any]] = []

    for xcol, ycol, description in pairs:
        if xcol not in df.columns or ycol not in df.columns:
            continue

        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        mask = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))

        n = int(mask.sum())
        if n < 3:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(x[mask].corr(y[mask], method="pearson"))
            spearman = float(x[mask].corr(y[mask], method="spearman"))

        rows.append(
            {
                "x": xcol,
                "y": ycol,
                "description": description,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    return pd.DataFrame(rows)


def bucket_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "joint_bucket" not in df.columns:
        return pd.DataFrame(columns=["joint_bucket", "count"])

    out = (
        df["joint_bucket"]
        .value_counts(dropna=False)
        .rename_axis("joint_bucket")
        .reset_index(name="count")
    )
    return out


def write_joint_outputs(
    *,
    outdir: str | Path,
    joined: pd.DataFrame,
    top: int = 50,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "joined": out / "guiding_beamlike_joined.csv",
        "bucket_counts": out / "guiding_beamlike_bucket_counts.csv",
        "correlations": out / "guiding_beamlike_correlations.csv",
        "both_positive": out / "both_positive_guiding_beamlike.csv",
        "guiding_positive_beam_negative": out / "guiding_positive_beam_negative.csv",
        "guiding_positive_beam_neutral": out / "guiding_positive_beam_neutral.csv",
        "beam_positive_guiding_nonpositive": out
        / "beam_positive_guiding_nonpositive.csv",
        "both_negative": out / "both_negative_guiding_beamlike.csv",
    }

    joined.to_csv(paths["joined"], index=False)
    bucket_counts(joined).to_csv(paths["bucket_counts"], index=False)
    compute_joint_correlations(joined).to_csv(paths["correlations"], index=False)

    if joined.empty:
        for key in [
            "both_positive",
            "guiding_positive_beam_negative",
            "guiding_positive_beam_neutral",
            "beam_positive_guiding_nonpositive",
            "both_negative",
        ]:
            joined.head(0).to_csv(paths[key], index=False)
        return paths

    both_positive = joined[
        joined["joint_bucket"] == "guiding_positive__beam_positive"
    ].copy()
    both_positive = both_positive.sort_values(
        "joint_positive_score",
        ascending=False,
    ).head(top)

    guiding_positive_beam_negative = joined[
        joined["joint_bucket"] == "guiding_positive__beam_negative"
    ].copy()
    guiding_positive_beam_negative = guiding_positive_beam_negative.sort_values(
        "guiding_final_score",
        ascending=False,
    ).head(top)

    guiding_positive_beam_neutral = joined[
        joined["joint_bucket"] == "guiding_positive__beam_neutral"
    ].copy()
    guiding_positive_beam_neutral = guiding_positive_beam_neutral.sort_values(
        "guiding_final_score",
        ascending=False,
    ).head(top)

    beam_positive_guiding_nonpositive = joined[
        (joined["beam_bucket"] == "positive") & (joined["guiding_bucket"] != "positive")
    ].copy()
    beam_positive_guiding_nonpositive = beam_positive_guiding_nonpositive.sort_values(
        "beam_beamlike_gain_score",
        ascending=False,
    ).head(top)

    both_negative = joined[
        joined["joint_bucket"] == "guiding_negative__beam_negative"
    ].copy()
    both_negative = both_negative.sort_values(
        "guiding_beam_alignment_factor",
        ascending=False,
    ).head(top)

    both_positive.to_csv(paths["both_positive"], index=False)
    guiding_positive_beam_negative.to_csv(
        paths["guiding_positive_beam_negative"],
        index=False,
    )
    guiding_positive_beam_neutral.to_csv(
        paths["guiding_positive_beam_neutral"],
        index=False,
    )
    beam_positive_guiding_nonpositive.to_csv(
        paths["beam_positive_guiding_nonpositive"],
        index=False,
    )
    both_negative.to_csv(paths["both_negative"], index=False)

    return paths
