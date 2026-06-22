#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _case_name_from_particle_analysis_dir(outdir: Path) -> str:
    if outdir.name == "particle_analysis":
        return outdir.parent.name
    return outdir.name


def _load_acceptance_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing acceptance CSV: {path}")

    df = pd.read_csv(path)

    required = {
        "theta_cut_mrad",
        "E_min_MeV",
        "accepted_charge_pC",
        "accepted_n_macroparticles",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["theta_cut_mrad"] = pd.to_numeric(df["theta_cut_mrad"], errors="coerce")
    df["E_min_MeV"] = pd.to_numeric(df["E_min_MeV"], errors="coerce")
    df["accepted_charge_pC"] = pd.to_numeric(df["accepted_charge_pC"], errors="coerce")
    df["accepted_n_macroparticles"] = pd.to_numeric(
        df["accepted_n_macroparticles"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "theta_cut_mrad",
            "E_min_MeV",
            "accepted_charge_pC",
            "accepted_n_macroparticles",
        ]
    )

    if df.empty:
        raise ValueError(f"No valid rows found in {path}")

    return df


def _maybe_load_summary(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    row = df.iloc[-1].to_dict()
    out: dict[str, float | int | str] = {}

    for key, value in row.items():
        try:
            out[key] = float(value)
        except Exception:
            out[key] = str(value)

    return out


def _fmt(value: object, unit: str = "") -> str:
    try:
        f = float(value)
    except Exception:
        return "nan"

    if not np.isfinite(f):
        return "nan"

    return f"{f:.3g}{unit}"


def _summary_annotation(summary: dict[str, object]) -> str:
    if not summary:
        return ""

    fields = [
        ("n_hot", "n_macroparticles_hot", ""),
        ("Q_hot", "charge_hot_pC", " pC"),
        ("E95_hot", "E95_hot_MeV", " MeV"),
        ("theta_r rms", "theta_rms_mrad", " mrad"),
        ("theta_r p95", "theta_r_p95_mrad", " mrad"),
        ("emit_n geom", "emit_geom_norm_mm_mrad", " mm mrad"),
        ("TQ score", "beam_transverse_quality_score", ""),
    ]

    lines = []
    for label, key, unit in fields:
        if key in summary:
            lines.append(f"{label}: {_fmt(summary[key], unit)}")

    return "\n".join(lines)


def save_acceptance_heatmap(
    df: pd.DataFrame,
    *,
    path: Path,
    case_name: str,
    summary: dict[str, object],
    value_column: str = "accepted_charge_pC",
) -> Path:
    pivot = (
        df.pivot_table(
            index="E_min_MeV",
            columns="theta_cut_mrad",
            values=value_column,
            aggfunc="max",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    values = pivot.to_numpy(dtype=float)
    e_values = pivot.index.to_numpy(dtype=float)
    theta_values = pivot.columns.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
    )

    cbar = fig.colorbar(image, ax=ax)
    ylabel = "accepted charge [pC]"
    if value_column == "accepted_n_macroparticles":
        ylabel = "accepted macroparticles"
    cbar.set_label(ylabel)

    ax.set_xticks(np.arange(len(theta_values)))
    ax.set_xticklabels([f"{v:g}" for v in theta_values])
    ax.set_yticks(np.arange(len(e_values)))
    ax.set_yticklabels([f"{v:g}" for v in e_values])

    ax.set_xlabel("theta_r cut [mrad]")
    ax.set_ylabel("E_min [MeV]")
    ax.set_title(f"Acceptance map — {case_name}")

    for i, e_min in enumerate(e_values):
        for j, theta in enumerate(theta_values):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    annotation = _summary_annotation(summary)
    if annotation:
        ax.text(
            1.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return path


def save_charge_vs_theta(
    df: pd.DataFrame,
    *,
    path: Path,
    case_name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for e_min, group in sorted(df.groupby("E_min_MeV"), key=lambda item: item[0]):
        group = group.sort_values("theta_cut_mrad")
        ax.plot(
            group["theta_cut_mrad"],
            group["accepted_charge_pC"],
            marker="o",
            markersize=3,
            label=f"E ≥ {e_min:g} MeV",
        )

    ax.set_xlabel("theta_r cut [mrad]")
    ax.set_ylabel("accepted charge [pC]")
    ax.set_title(f"Accepted charge vs divergence cut — {case_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_charge_vs_energy(
    df: pd.DataFrame,
    *,
    path: Path,
    case_name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for theta, group in sorted(df.groupby("theta_cut_mrad"), key=lambda item: item[0]):
        group = group.sort_values("E_min_MeV")
        ax.plot(
            group["E_min_MeV"],
            group["accepted_charge_pC"],
            marker="o",
            markersize=3,
            label=f"theta ≤ {theta:g} mrad",
        )

    ax.set_xlabel("E_min [MeV]")
    ax.set_ylabel("accepted charge [pC]")
    ax.set_title(f"Accepted charge vs energy cut — {case_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_summary_bars(
    summary: dict[str, object],
    *,
    path: Path,
    case_name: str,
) -> Path | None:
    if not summary:
        return None

    keys = [
        "theta_rms_mrad",
        "theta_r_p95_mrad",
        "emit_geom_norm_mm_mrad",
        "beam_transverse_quality_score",
    ]

    labels = [
        "theta rms [mrad]",
        "theta p95 [mrad]",
        "emit geom [mm mrad]",
        "TQ score",
    ]

    values = []
    used_labels = []
    for key, label in zip(keys, labels):
        if key not in summary:
            continue
        try:
            value = float(summary[key])
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
            used_labels.append(label)

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(used_labels, values)
    ax.set_title(f"Reduced transverse metrics — {case_name}")
    ax.set_ylabel("value")
    ax.tick_params(axis="x", rotation=25)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_one_particle_analysis_dir(outdir: Path) -> list[Path]:
    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    case_name = _case_name_from_particle_analysis_dir(outdir)

    acceptance_csv = outdir / "particle_acceptance_curves.csv"
    summary_csv = outdir / "particle_summary.csv"

    df = _load_acceptance_csv(acceptance_csv)
    summary = _maybe_load_summary(summary_csv)

    outputs: list[Path] = []

    outputs.append(
        save_acceptance_heatmap(
            df,
            path=plots_dir / "particle_acceptance_heatmap_charge_pC.png",
            case_name=case_name,
            summary=summary,
            value_column="accepted_charge_pC",
        )
    )

    outputs.append(
        save_acceptance_heatmap(
            df,
            path=plots_dir / "particle_acceptance_heatmap_n_macroparticles.png",
            case_name=case_name,
            summary=summary,
            value_column="accepted_n_macroparticles",
        )
    )

    outputs.append(
        save_charge_vs_theta(
            df,
            path=plots_dir / "particle_acceptance_charge_vs_theta.png",
            case_name=case_name,
        )
    )

    outputs.append(
        save_charge_vs_energy(
            df,
            path=plots_dir / "particle_acceptance_charge_vs_energy.png",
            case_name=case_name,
        )
    )

    summary_plot = save_summary_bars(
        summary,
        path=plots_dir / "particle_transverse_summary_bars.png",
        case_name=case_name,
    )
    if summary_plot is not None:
        outputs.append(summary_plot)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reduced particle acceptance diagnostics from existing CSV files. "
            "Does not require raw openPMD/HDF5 particle dumps."
        )
    )
    parser.add_argument(
        "--particle-analysis-dir",
        action="append",
        required=True,
        help=("Path to CASE/particle_analysis. Can be passed multiple times."),
    )

    args = parser.parse_args()

    total = 0
    for raw in args.particle_analysis_dir:
        outdir = Path(raw)
        print(f"[CASE] {outdir}")
        paths = plot_one_particle_analysis_dir(outdir)
        for path in paths:
            print(f"[OK] wrote {path}")
        total += len(paths)

    print(f"[DONE] wrote {total} plots")


if __name__ == "__main__":
    main()
