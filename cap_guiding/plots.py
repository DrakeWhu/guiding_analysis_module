from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import first_valid_laser_index


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def draw_plateau_window(
    ax,
    plateau_window_mm: tuple[float, float] | None,
    labels: bool = False,
) -> None:
    """Draw plateau start/end vertical markers on an existing matplotlib axis."""
    if plateau_window_mm is None:
        return

    plateau_start_mm, plateau_end_mm = plateau_window_mm

    ax.axvline(
        plateau_start_mm,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.axvline(
        plateau_end_mm,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )

    if labels:
        ax.text(
            plateau_start_mm,
            0.02,
            " plateau start",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=8,
        )
        ax.text(
            plateau_end_mm,
            0.02,
            " plateau end",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=8,
        )


def infer_plateau_window_mm_from_name(name: str) -> tuple[float, float] | None:
    """Infer plateau start/end positions in mm from a case/output name.

    Convention used in current campaigns:
    - ramp-up always starts at z = 0 mm and ends at z = 5 mm
    - plateau length is encoded as L5mm, L10mm, L25mm, etc.
    - therefore:
        plateau_start_mm = 5.0
        plateau_end_mm = 5.0 + plateau_length_mm
    """
    m = re.search(r"_L(\d+(?:\.\d+)?)mm_", name)
    if m is None:
        m = re.search(r"_L(\d+(?:\.\d+)?)mm($|_)", name)

    if m is None:
        return None

    plateau_length_mm = float(m.group(1))
    plateau_start_mm = 5.0
    plateau_end_mm = plateau_start_mm + plateau_length_mm
    return plateau_start_mm, plateau_end_mm


def draw_plateau_window(
    ax,
    plateau_window_mm: tuple[float, float] | None,
    labels: bool = False,
) -> None:
    """Draw vertical markers for plateau start/end on an existing axis."""
    if plateau_window_mm is None:
        return

    plateau_start_mm, plateau_end_mm = plateau_window_mm

    ax.axvline(
        plateau_start_mm,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.axvline(
        plateau_end_mm,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )

    if labels:
        ax.text(
            plateau_start_mm,
            0.02,
            " plateau start",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=8,
        )
        ax.text(
            plateau_end_mm,
            0.02,
            " plateau end",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=8,
        )


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

    plateau_window_mm = infer_plateau_window_mm_from_name(outdir.name)

    propagation_mm = _arr(rows, "propagation_mm")
    z_peak = _arr(rows, "z_peak_um")
    waist = _arr(rows, "waist_um")
    front_margin = _arr(rows, "front_margin_um")
    peak_I = _arr(rows, "peak_I_proxy")
    energy = _arr(rows, "energy_proxy")
    Ez_abs = _arr(rows, "Ez_wake_absmax")

    has_a0 = "a0_peak" in rows[0]
    if has_a0:
        a0_peak = _arr(rows, "a0_peak")
    else:
        a0_peak = None
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

    if has_a0:
        a0_norm = np.full_like(a0_peak, np.nan, dtype=float)
        if np.isfinite(a0_peak[ref]) and a0_peak[ref] > 0.0:
            a0_norm[valid] = a0_peak[valid] / a0_peak[ref]
        else:
            a0_norm[:] = np.nan
    else:
        a0_norm = None

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

    def save_plot(y, ylabel, name, hlines=None):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(propagation_mm, y, marker="o", markersize=3)
        add_breakdown_marker(ax)
        draw_plateau_window(ax, plateau_window_mm, labels=True)

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
    if has_a0:
        save_plot(a0_peak, "laser peak a0", "laser_a0_peak.png")
        save_plot(
            a0_norm,
            f"peak a0 / dump {ref_iteration}",
            "laser_a0_norm.png",
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
        (
            a0_norm if has_a0 else peak_I_norm,
            f"a0 / dump {ref_iteration}"
            if has_a0
            else f"peak I / dump {ref_iteration}",
        ),
        (energy_norm, f"energy / dump {ref_iteration}"),
        (Ez_abs / 1.0e9, "max |Ez wake| [GV/m]"),
        (z_Ez_rel, "z(Ez absmax) - z_peak [um]"),
    ]

    for ax, (y, label) in zip(axs, panels):
        ax.plot(propagation_mm, y, marker="o", markersize=3)
        add_breakdown_marker(ax)
        draw_plateau_window(ax, plateau_window_mm, labels=False)
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
    plateau_window_mm: tuple[float, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = wide["propagation_mm"]

    for col, label in ycols:
        ax.plot(x, wide[col], marker="o", markersize=3, linewidth=1.2, label=label)

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1)

    draw_plateau_window(ax, plateau_window_mm, labels=True)

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

    plateau_window_mm = wide.attrs.get("plateau_window_mm")

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

    has_a0_triplet = all(f"a0_peak_{c}" in wide.columns for c in case_order)

    if has_a0_triplet:
        save_triplet_line_plot(
            wide,
            [(f"a0_peak_{c}", c) for c in case_order],
            "peak a0",
            "Peak a0 comparison",
            plots / "a0_peak_comparison.png",
        )

        save_triplet_line_plot(
            wide,
            [
                ("a0_channel_over_vacuum", "channel/vacuum"),
                ("a0_uniform_over_vacuum", "uniform/vacuum"),
                ("a0_channel_over_uniform", "channel/uniform"),
            ],
            "a0 ratio",
            "Peak a0 ratios",
            plots / "a0_ratios.png",
            hline=1.0,
        )
    else:
        has_a0_triplet = False

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    panels = [
        (
            [(f"waist_um_{c}", c) for c in case_order],
            "waist RMS [µm]",
            "Optical confinement",
            None,
        ),
        (
            (
                [(f"a0_norm_{c}", c) for c in case_order]
                if has_a0_triplet
                else [(f"peak_I_norm_{c}", c) for c in case_order]
            ),
            "a0 / first valid dump" if has_a0_triplet else "peak I / first valid dump",
            "Peak a0" if has_a0_triplet else "Peak intensity",
            None,
        ),
        (
            [("waist_channel_over_uniform", "waist channel/uniform")],
            "ratio",
            "Channel vs uniform: waist",
            1.0,
        ),
        (
            (
                [("a0_channel_over_uniform", "a0 channel/uniform")]
                if has_a0_triplet
                else [("peakI_channel_over_uniform", "peak I channel/uniform")]
            ),
            "ratio",
            "Channel vs uniform: a0"
            if has_a0_triplet
            else "Channel vs uniform: peak I",
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

        draw_plateau_window(
            ax,
            wide.attrs.get("plateau_window_mm"),
            labels=False,
        )

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
