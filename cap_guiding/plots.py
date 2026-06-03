from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import first_valid_laser_index


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _detect_breakdown(
    rows: list[dict[str, Any]],
) -> tuple[int | None, float | None, int | None]:
    propagation_mm = _arr(rows, "propagation_mm")
    waist = _arr(rows, "waist_um")
    Ez_abs = _arr(rows, "Ez_wake_absmax")

    ref = first_valid_laser_index(rows)
    waist_ref = waist[ref]

    plateau_mask = (
        np.isfinite(Ez_abs) & (propagation_mm >= 0.5) & (propagation_mm <= 3.0)
    )

    if np.count_nonzero(plateau_mask) < 3:
        return None, None, None

    Ez_plateau = np.nanmedian(Ez_abs[plateau_mask])
    candidates = np.where(
        (propagation_mm > 1.0)
        & np.isfinite(Ez_abs)
        & np.isfinite(waist)
        & (Ez_abs < 0.70 * Ez_plateau)
        & (waist > 1.10 * waist_ref)
    )[0]

    if candidates.size == 0:
        return None, None, None

    idx = int(candidates[0])
    return idx, float(propagation_mm[idx]), int(rows[idx]["iteration"])


def save_case_plots(rows: list[dict[str, Any]], outdir: str | Path) -> None:
    outdir = Path(outdir)
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    propagation_mm = _arr(rows, "propagation_mm")
    z_peak = _arr(rows, "z_peak_um")
    waist = _arr(rows, "waist_um")
    front_margin = _arr(rows, "front_margin_um")
    peak_I = _arr(rows, "peak_I_proxy")
    energy = _arr(rows, "energy_proxy")
    Ez_abs = _arr(rows, "Ez_wake_absmax")
    z_Ez_rel = _arr(rows, "z_Ez_absmax_rel_um")

    ref = first_valid_laser_index(rows)
    ref_iteration = int(rows[ref]["iteration"])

    valid = (
        np.isfinite(peak_I)
        & np.isfinite(energy)
        & np.isfinite(waist)
        & (peak_I > 0.0)
        & (energy > 0.0)
    )

    peak_I_norm = np.full_like(peak_I, np.nan, dtype=float)
    energy_norm = np.full_like(energy, np.nan, dtype=float)
    peak_I_norm[valid] = peak_I[valid] / peak_I[ref]
    energy_norm[valid] = energy[valid] / energy[ref]

    _, breakdown_mm, breakdown_iteration = _detect_breakdown(rows)
    if breakdown_mm is not None:
        print(
            "[INFO] tentative breakdown: "
            f"iteration={breakdown_iteration}, propagation={breakdown_mm:.3f} mm"
        )
    else:
        print("[INFO] no tentative breakdown detected with current criterion")

    def add_breakdown_marker(ax):
        if breakdown_mm is not None:
            ax.axvline(breakdown_mm, linestyle="--", linewidth=1)
            ax.text(
                breakdown_mm,
                0.98,
                " breakdown?",
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="left",
                fontsize=8,
            )

    def save_plot(y, ylabel: str, name: str, hlines=None):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(propagation_mm, y, marker="o", markersize=3)
        add_breakdown_marker(ax)

        if hlines:
            for value, label in hlines:
                ax.axhline(value, linestyle="--", linewidth=1)
                ax.text(
                    propagation_mm[-1],
                    value,
                    f" {label}",
                    va="bottom",
                    ha="left",
                    fontsize=8,
                )

        ax.set_xlabel("propagation distance [mm]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()

        path = plots / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[OK] wrote {path}")

    save_plot(z_peak, "laser z_peak [um]", "laser_z_peak.png")
    save_plot(
        front_margin,
        "front margin zmax - z_peak [um]",
        "laser_front_margin.png",
        hlines=[(20.0, "20 um"), (30.0, "30 um"), (50.0, "50 um")],
    )
    save_plot(waist, "laser waist rms [um]", "laser_waist_rms.png")
    save_plot(
        peak_I_norm,
        f"peak I proxy / dump {ref_iteration}",
        "laser_peak_I_proxy_norm.png",
    )
    save_plot(
        energy_norm,
        f"energy proxy / dump {ref_iteration}",
        "laser_energy_proxy_norm.png",
    )
    save_plot(Ez_abs / 1.0e9, "max |Ez wake| [GV/m]", "wake_Ez_absmax_GVm.png")
    save_plot(
        z_Ez_rel,
        "z(Ez absmax) - z_peak [um]",
        "wake_Ez_absmax_relative_position.png",
    )

    fig, axs = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    axs = axs.ravel()

    panels = [
        (front_margin, "front margin [um]"),
        (waist, "laser waist rms [um]"),
        (peak_I_norm, f"peak I / dump {ref_iteration}"),
        (energy_norm, f"energy / dump {ref_iteration}"),
        (Ez_abs / 1.0e9, "max |Ez wake| [GV/m]"),
        (z_Ez_rel, "z(Ez absmax) - z_peak [um]"),
    ]

    for ax, (y, label) in zip(axs, panels):
        ax.plot(propagation_mm, y, marker="o", markersize=3)
        add_breakdown_marker(ax)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    axs[-2].set_xlabel("propagation distance [mm]")
    axs[-1].set_xlabel("propagation distance [mm]")

    fig.suptitle("Capillary guiding summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = plots / "guiding_summary_multipanel.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {path}")


def save_triplet_line_plot(
    wide,
    ycols: list[tuple[str, str]],
    ylabel: str,
    title: str,
    path: str | Path,
    hline: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = wide["propagation_mm"]

    for col, label in ycols:
        ax.plot(x, wide[col], marker="o", markersize=3, linewidth=1.2, label=label)

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1)

    ax.set_xlabel("propagation distance [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"[OK] wrote {path}")


def save_triplet_plots(wide, outdir: str | Path) -> None:
    outdir = Path(outdir)
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    case_order = ["channel", "uniform", "vacuum"]

    save_triplet_line_plot(
        wide,
        [(f"waist_um_{c}", c) for c in case_order],
        "laser waist RMS [µm]",
        "Laser waist comparison",
        plots / "waist_comparison.png",
    )

    save_triplet_line_plot(
        wide,
        [(f"peak_I_norm_{c}", c) for c in case_order],
        "peak I proxy / first valid dump",
        "Peak intensity proxy comparison",
        plots / "peakI_norm_comparison.png",
    )

    save_triplet_line_plot(
        wide,
        [(f"energy_norm_{c}", c) for c in case_order],
        "energy proxy / first valid dump",
        "Laser energy proxy comparison",
        plots / "energy_norm_comparison.png",
    )

    save_triplet_line_plot(
        wide,
        [(f"Ez_wake_absmax_GVm_{c}", c) for c in case_order],
        "max |Ez wake| [GV/m]",
        "Wake-field amplitude comparison",
        plots / "Ez_wake_absmax_comparison.png",
    )

    save_triplet_line_plot(
        wide,
        [
            ("waist_channel_over_vacuum", "channel/vacuum"),
            ("waist_uniform_over_vacuum", "uniform/vacuum"),
            ("waist_channel_over_uniform", "channel/uniform"),
        ],
        "waist ratio",
        "Waist ratios; lower is stronger optical confinement",
        plots / "waist_ratios.png",
        hline=1.0,
    )

    save_triplet_line_plot(
        wide,
        [
            ("peakI_channel_over_vacuum", "channel/vacuum"),
            ("peakI_uniform_over_vacuum", "uniform/vacuum"),
            ("peakI_channel_over_uniform", "channel/uniform"),
        ],
        "peak I ratio",
        "Peak-intensity ratios; higher means stronger peak preservation/focusing",
        plots / "peakI_ratios.png",
        hline=1.0,
    )

    save_triplet_line_plot(
        wide,
        [
            ("energy_channel_over_vacuum", "channel/vacuum"),
            ("energy_uniform_over_vacuum", "uniform/vacuum"),
            ("energy_channel_over_uniform", "channel/uniform"),
        ],
        "energy proxy ratio",
        "Energy-proxy ratios; interpret with depletion/transfer in mind",
        plots / "energy_ratios.png",
        hline=1.0,
    )

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    panels = [
        (
            [(f"waist_um_{c}", c) for c in case_order],
            "waist RMS [µm]",
            "Optical confinement",
            None,
        ),
        (
            [(f"peak_I_norm_{c}", c) for c in case_order],
            "peak I / first valid dump",
            "Peak intensity",
            None,
        ),
        (
            [("waist_channel_over_uniform", "waist channel/uniform")],
            "ratio",
            "Channel vs uniform: waist",
            1.0,
        ),
        (
            [("peakI_channel_over_uniform", "peak I channel/uniform")],
            "ratio",
            "Channel vs uniform: peak I",
            1.0,
        ),
    ]

    for ax, (cols, ylabel, title, hline) in zip(axs.ravel(), panels):
        for col, label in cols:
            ax.plot(
                wide["propagation_mm"],
                wide[col],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=label,
            )

        if hline is not None:
            ax.axhline(hline, linestyle="--", linewidth=1)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axs[1, 0].set_xlabel("propagation distance [mm]")
    axs[1, 1].set_xlabel("propagation distance [mm]")

    fig.suptitle("Guiding triplet comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = plots / "guiding_triplet_teaser_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"[OK] wrote {path}")
