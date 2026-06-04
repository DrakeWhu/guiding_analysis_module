from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from .case_metadata import infer_plateau_window_mm_from_sources


CASE_ORDER = ["channel", "uniform", "vacuum"]

REQUIRED_COLUMNS = [
    "iteration",
    "propagation_mm",
    "waist_um",
    "peak_I_proxy",
    "energy_proxy",
    "Ez_wake_absmax",
    "front_margin_um",
]

OPTIONAL_COLUMNS = [
    "Eperp_peak_Vm",
    "a0_peak",
]


def read_case_csv(path: str | Path, case_type: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["case_type"] = case_type
    df["source_csv"] = str(path)

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def first_valid_index(df: pd.DataFrame) -> int:
    valid = (
        np.isfinite(df["peak_I_proxy"].to_numpy(float))
        & np.isfinite(df["energy_proxy"].to_numpy(float))
        & np.isfinite(df["waist_um"].to_numpy(float))
        & (df["peak_I_proxy"].to_numpy(float) > 0.0)
        & (df["energy_proxy"].to_numpy(float) > 0.0)
    )

    if not np.any(valid):
        raise ValueError("No valid laser dump found in one case CSV")

    return int(np.where(valid)[0][0])


def add_case_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("iteration").reset_index(drop=True).copy()

    i0 = first_valid_index(df)

    peak0 = float(df.loc[i0, "peak_I_proxy"])
    energy0 = float(df.loc[i0, "energy_proxy"])
    waist0 = float(df.loc[i0, "waist_um"])

    df["peak_I_norm"] = df["peak_I_proxy"] / peak0
    df["energy_norm"] = df["energy_proxy"] / energy0
    df["waist_norm"] = df["waist_um"] / waist0
    df["Ez_wake_absmax_GVm"] = df["Ez_wake_absmax"] / 1.0e9

    if "a0_peak" in df.columns:
        a0_0 = float(df.loc[i0, "a0_peak"])
        if np.isfinite(a0_0) and a0_0 > 0.0:
            df["a0_norm"] = df["a0_peak"] / a0_0
        else:
            df["a0_norm"] = np.nan

    df["ref_iteration"] = int(df.loc[i0, "iteration"])

    return df


def load_triplet_csvs(
    channel_csv: str | Path,
    uniform_csv: str | Path,
    vacuum_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    return {
        "channel": add_case_normalizations(read_case_csv(channel_csv, "channel")),
        "uniform": add_case_normalizations(read_case_csv(uniform_csv, "uniform")),
        "vacuum": add_case_normalizations(read_case_csv(vacuum_csv, "vacuum")),
    }


def align_common_iterations(cases: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    missing_cases = [case for case in CASE_ORDER if case not in cases]
    if missing_cases:
        raise ValueError(f"Missing cases: {missing_cases}")

    common: set[int] | None = None

    for case in CASE_ORDER:
        df = cases[case]
        its = set(df["iteration"].dropna().astype(int).tolist())
        common = its if common is None else common & its

    if not common:
        raise ValueError("No common iterations across channel/uniform/vacuum")

    # Semantic rule: channel is the limiting/reference case.
    channel_last = int(cases["channel"]["iteration"].max())
    common_sorted = sorted(int(it) for it in common if int(it) <= channel_last)

    aligned: dict[str, pd.DataFrame] = {}

    for case in CASE_ORDER:
        df = cases[case]
        d = df[df["iteration"].astype(int).isin(common_sorted)].copy()
        d = d.sort_values("iteration").reset_index(drop=True)
        aligned[case] = d

    lengths = {case: len(df) for case, df in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Internal alignment error: {lengths}")

    return aligned


def build_long(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([aligned[c] for c in CASE_ORDER], ignore_index=True)


def build_wide(aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = aligned["channel"][["iteration", "propagation_mm"]].copy()

    metric_cols = [
        "waist_um",
        "waist_norm",
        "peak_I_proxy",
        "peak_I_norm",
        "energy_proxy",
        "energy_norm",
        "Ez_wake_absmax",
        "Ez_wake_absmax_GVm",
        "front_margin_um",
    ]

    for optional_col in ["Eperp_peak_Vm", "a0_peak", "a0_norm"]:
        if all(optional_col in aligned[case].columns for case in CASE_ORDER):
            metric_cols.append(optional_col)

    wide = base

    for case in CASE_ORDER:
        d = aligned[case][["iteration", *metric_cols]].copy()
        d = d.rename(columns={col: f"{col}_{case}" for col in metric_cols})
        wide = wide.merge(d, on="iteration", how="inner")

    ratio_pairs = [
        ("channel", "vacuum"),
        ("uniform", "vacuum"),
        ("channel", "uniform"),
    ]

    for a, b in ratio_pairs:
        wide[f"waist_{a}_over_{b}"] = wide[f"waist_um_{a}"] / wide[f"waist_um_{b}"]
        wide[f"peakI_{a}_over_{b}"] = (
            wide[f"peak_I_proxy_{a}"] / wide[f"peak_I_proxy_{b}"]
        )
        wide[f"energy_{a}_over_{b}"] = (
            wide[f"energy_proxy_{a}"] / wide[f"energy_proxy_{b}"]
        )
        wide[f"Ezabs_{a}_over_{b}"] = (
            wide[f"Ez_wake_absmax_{a}"] / wide[f"Ez_wake_absmax_{b}"]
        )
        if f"a0_peak_{a}" in wide.columns and f"a0_peak_{b}" in wide.columns:
            wide[f"a0_{a}_over_{b}"] = wide[f"a0_peak_{a}"] / wide[f"a0_peak_{b}"]

    return wide


def late_mask(wide: pd.DataFrame, late_fraction: float) -> pd.Series:
    if not (0.0 < late_fraction <= 1.0):
        raise ValueError("late_fraction must be in (0, 1]")

    n = len(wide)
    n_late = max(1, int(np.ceil(late_fraction * n)))

    mask = pd.Series(False, index=wide.index)
    mask.iloc[-n_late:] = True

    return mask


def nanmedian(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)

    if np.all(~np.isfinite(arr)):
        return float("nan")

    return float(np.nanmedian(arr))


def infer_plateau_window_mm_from_aligned(
    aligned: dict[str, pd.DataFrame],
) -> tuple[float, float] | None:
    """Infer plateau window from source_csv paths in aligned case dataframes."""
    sources: list[str] = []

    for case in CASE_ORDER:
        df = aligned.get(case)
        if df is None or "source_csv" not in df.columns or df.empty:
            continue
        sources.append(str(df["source_csv"].iloc[0]))

    return infer_plateau_window_mm_from_sources(sources)


def build_late_summary(wide: pd.DataFrame, late_fraction: float) -> pd.DataFrame:
    mask = late_mask(wide, late_fraction)
    late = wide.loc[mask].copy()

    rows: list[dict[str, Any]] = []

    for case in CASE_ORDER:
        row = {
            "case_type": case,
            "n_common_dumps": len(wide),
            "n_late_dumps": len(late),
            "late_fraction": late_fraction,
            "propagation_start_mm": float(wide["propagation_mm"].iloc[0]),
            "propagation_end_mm": float(wide["propagation_mm"].iloc[-1]),
            "late_start_mm": float(late["propagation_mm"].iloc[0]),
            "late_end_mm": float(late["propagation_mm"].iloc[-1]),
            "waist_late_median_um": nanmedian(late[f"waist_um_{case}"]),
            "waist_norm_late_median": nanmedian(late[f"waist_norm_{case}"]),
            "peakI_norm_late_median": nanmedian(late[f"peak_I_norm_{case}"]),
            "energy_norm_late_median": nanmedian(late[f"energy_norm_{case}"]),
            "Ezabs_late_median_GVm": nanmedian(late[f"Ez_wake_absmax_GVm_{case}"]),
            "front_margin_late_median_um": nanmedian(late[f"front_margin_um_{case}"]),
        }

        if f"a0_peak_{case}" in late.columns:
            row["a0_peak_late_median"] = nanmedian(late[f"a0_peak_{case}"])

        if f"a0_norm_{case}" in late.columns:
            row["a0_norm_late_median"] = nanmedian(late[f"a0_norm_{case}"])

        rows.append(row)

    return pd.DataFrame(rows)


def build_late_ratios(wide: pd.DataFrame, late_fraction: float) -> pd.DataFrame:
    mask = late_mask(wide, late_fraction)
    late = wide.loc[mask].copy()

    ratio_cols = [
        "waist_channel_over_vacuum",
        "waist_uniform_over_vacuum",
        "waist_channel_over_uniform",
        "peakI_channel_over_vacuum",
        "peakI_uniform_over_vacuum",
        "peakI_channel_over_uniform",
        "energy_channel_over_vacuum",
        "energy_uniform_over_vacuum",
        "energy_channel_over_uniform",
        "Ezabs_channel_over_vacuum",
        "Ezabs_uniform_over_vacuum",
        "Ezabs_channel_over_uniform",
    ]

    row: dict[str, Any] = {
        "late_fraction": late_fraction,
        "n_late_dumps": len(late),
    }

    for col in ratio_cols:
        row[f"{col}_late_median"] = nanmedian(late[col])

    for col in [
        "a0_channel_over_vacuum",
        "a0_uniform_over_vacuum",
        "a0_channel_over_uniform",
    ]:
        if col in late.columns:
            row[f"{col}_late_median"] = nanmedian(late[col])

    row["optical_guiding_peakI_gain_ch_over_uni"] = row[
        "peakI_channel_over_uniform_late_median"
    ]

    row["optical_guiding_waist_reduction_ch_over_uni"] = (
        1.0 - row["waist_channel_over_uniform_late_median"]
    )

    return pd.DataFrame([row])


def build_triplet_tables(
    channel_csv: str | Path,
    uniform_csv: str | Path,
    vacuum_csv: str | Path,
    label: str = "triplet",
    late_fraction: float = 1.0 / 3.0,
) -> dict[str, pd.DataFrame]:
    cases = load_triplet_csvs(
        channel_csv=channel_csv,
        uniform_csv=uniform_csv,
        vacuum_csv=vacuum_csv,
    )

    aligned = align_common_iterations(cases)

    plateau_window_mm = infer_plateau_window_mm_from_aligned(aligned)

    long = build_long(aligned)
    wide = build_wide(aligned)

    if plateau_window_mm is not None:
        plateau_start_mm, plateau_end_mm = plateau_window_mm
        wide.attrs["plateau_window_mm"] = plateau_window_mm
        wide["plateau_start_mm"] = plateau_start_mm
        wide["plateau_end_mm"] = plateau_end_mm
        long.attrs["plateau_window_mm"] = plateau_window_mm

    late_summary = build_late_summary(wide, late_fraction)
    late_ratios = build_late_ratios(wide, late_fraction)

    if plateau_window_mm is not None:
        for df in (late_summary, late_ratios):
            df.attrs["plateau_window_mm"] = plateau_window_mm
            df["plateau_start_mm"] = plateau_window_mm[0]
            df["plateau_end_mm"] = plateau_window_mm[1]

    for df in (long, wide, late_summary, late_ratios):
        df.insert(0, "triplet", label)

    return {
        "long": long,
        "wide": wide,
        "late_summary": late_summary,
        "late_ratios": late_ratios,
    }


def write_triplet_tables(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "long": outdir / "guiding_triplet_long.csv",
        "wide": outdir / "guiding_triplet_wide.csv",
        "late_summary": outdir / "guiding_triplet_late_summary.csv",
        "late_ratios": outdir / "guiding_triplet_late_ratios.csv",
    }

    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
        print(f"[OK] wrote {path}")

    return paths
