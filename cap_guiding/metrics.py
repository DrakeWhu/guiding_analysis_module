from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .openpmd_io import open_series, get_iterations, read_field_rz

E_CHARGE_C = 1.602176634e-19  # elementary charge in Coulombs
M_E_KG = 9.1093837015e-31  # electron mass in kg
C_M_PER_S = 299792458.0  # speed of light in m/s


def moving_average(y: np.ndarray, window_cells: int) -> np.ndarray:
    if window_cells <= 1:
        return y
    kernel = np.ones(window_cells, dtype=float) / window_cells
    return np.convolve(y, kernel, mode="same")


def eperp_peak_to_a0(Eperp_peak_Vm: float, lambda0_m: float) -> float:
    """Convert peak transverse electric field to normalized vector potential a0.

    a0 = e E0 / (m_e c omega0), with omega0 = 2*pi*c/lambda0.
    """
    if not np.isfinite(Eperp_peak_Vm) or Eperp_peak_Vm <= 0.0:
        return float("nan")

    if not np.isfinite(lambda0_m) or lambda0_m <= 0.0:
        return float("nan")

    omega0 = 2.0 * math.pi * C_M_PER_S / lambda0_m
    return float(E_CHARGE_C * Eperp_peak_Vm / (M_E_KG * C_M_PER_S * omega0))


def weighted_laser_metrics(
    Ex_rz: np.ndarray,
    Ey_rz: np.ndarray,
    r_um: np.ndarray,
    z_um: np.ndarray,
    smooth_um: float = 2.0,
    lambda0_m: float = 0.8e-6,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute the laser guiding metrics used in the original F20 analysis.

    Inputs must be positive-r arrays with shape [r, z].
    """
    I = Ex_rz**2 + Ey_rz**2

    r_m = r_um * 1.0e-6
    z_m = z_um * 1.0e-6

    # Cylindrical radial weight; the constant 2*pi cancels in ratios.
    radial_weight = np.maximum(r_m, 0.0)

    I_z = np.sum(I * radial_weight[:, None], axis=0)

    dz_um = float(np.median(np.diff(z_um)))
    smooth_cells = max(1, int(round(smooth_um / dz_um)))
    I_z_smooth = moving_average(I_z, smooth_cells)

    j_peak = int(np.argmax(I_z_smooth))
    z_peak_um = float(z_um[j_peak])

    z_window_um = 5.0
    z_mask = np.abs(z_um - z_peak_um) <= z_window_um

    if not np.any(z_mask):
        z_mask = np.zeros_like(z_um, dtype=bool)
        z_mask[j_peak] = True

    profile = np.sum(I[:, z_mask], axis=1)
    denom = np.sum(profile * radial_weight)

    if denom > 0:
        r2_mean = np.sum((r_m**2) * profile * radial_weight) / denom
        waist_um = float(np.sqrt(2.0 * r2_mean) * 1.0e6)
    else:
        waist_um = float("nan")

    dr_m = float(np.median(np.diff(r_m)))
    dz_m = float(np.median(np.diff(z_m)))
    energy_proxy = float(np.sum(I * radial_weight[:, None]) * dr_m * dz_m)

    peak_I_proxy = float(np.max(I))
    Eperp_peak_Vm = float(np.sqrt(peak_I_proxy))
    a0_peak = eperp_peak_to_a0(Eperp_peak_Vm, lambda0_m=lambda0_m)

    front_margin_um = float(z_um.max() - z_peak_um)
    back_margin_um = float(z_peak_um - z_um.min())

    return {
        "z_peak_um": z_peak_um,
        "front_margin_um": front_margin_um,
        "back_margin_um": back_margin_um,
        "waist_um": waist_um,
        "peak_I_proxy": peak_I_proxy,
        "Eperp_peak_Vm": Eperp_peak_Vm,
        "a0_peak": a0_peak,
        "energy_proxy": energy_proxy,
    }, I_z_smooth


def wake_metrics(
    Ez_rz: np.ndarray,
    r_um: np.ndarray,
    z_um: np.ndarray,
    z_peak_um: float,
    wake_behind_um: float = 120.0,
    wake_gap_um: float = 5.0,
) -> dict[str, float]:
    """Compute on-axis wake metrics behind the laser peak."""
    i_axis = int(np.argmin(np.abs(r_um)))
    Ez_axis = Ez_rz[i_axis, :]

    z1 = z_peak_um - wake_behind_um
    z2 = z_peak_um - wake_gap_um
    mask = (z_um >= z1) & (z_um <= z2)

    if not np.any(mask):
        return {
            "Ez_wake_max": float("nan"),
            "Ez_wake_min": float("nan"),
            "Ez_wake_absmax": float("nan"),
            "Ez_wake_rms": float("nan"),
            "z_Ez_absmax_um": float("nan"),
            "z_Ez_absmax_rel_um": float("nan"),
        }

    Ez_wake = Ez_axis[mask]
    z_wake = z_um[mask]
    k = int(np.argmax(np.abs(Ez_wake)))
    z_Ez_absmax_um = float(z_wake[k])

    return {
        "Ez_wake_max": float(np.max(Ez_wake)),
        "Ez_wake_min": float(np.min(Ez_wake)),
        "Ez_wake_absmax": float(np.max(np.abs(Ez_wake))),
        "Ez_wake_rms": float(np.sqrt(np.mean(Ez_wake**2))),
        "z_Ez_absmax_um": z_Ez_absmax_um,
        "z_Ez_absmax_rel_um": z_Ez_absmax_um - float(z_peak_um),
    }


def valid_laser_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    z_peak = np.array([r["z_peak_um"] for r in rows], dtype=float)
    waist = np.array([r["waist_um"] for r in rows], dtype=float)
    peak_I = np.array([r["peak_I_proxy"] for r in rows], dtype=float)
    energy = np.array([r["energy_proxy"] for r in rows], dtype=float)

    return (
        np.isfinite(z_peak)
        & np.isfinite(waist)
        & np.isfinite(peak_I)
        & np.isfinite(energy)
        & (peak_I > 0.0)
        & (energy > 0.0)
    )


def first_valid_laser_index(rows: list[dict[str, Any]]) -> int:
    valid = valid_laser_mask(rows)
    if not np.any(valid):
        raise RuntimeError(
            "No valid laser dumps found: peak_I/energy are zero or NaN everywhere."
        )
    return int(np.where(valid)[0][0])


def add_propagation_columns(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to annotate")

    z_max = np.array([r["z_max_um"] for r in rows], dtype=float)
    z_min = np.array([r["z_min_um"] for r in rows], dtype=float)
    z_peak = np.array([r["z_peak_um"] for r in rows], dtype=float)

    # Monotonic moving-window coordinate. Do not use z_peak as propagation axis.
    propagation_mm = (z_max - z_max[0]) * 1.0e-3
    z_peak_relative_um = z_peak - z_min

    for row, prop, zrel in zip(rows, propagation_mm, z_peak_relative_um):
        row["propagation_mm"] = float(prop)
        row["z_peak_relative_um"] = float(zrel)


def compute_case_rows(
    diag: str | Path,
    stride: int = 1,
    smooth_um: float = 2.0,
    wake_behind_um: float = 120.0,
    wake_gap_um: float = 5.0,
    lambda0_m: float = 0.8e-6,
) -> list[dict[str, Any]]:
    """Read one WarpX RZ diagnostic and compute guiding metrics for each dump."""
    ts = open_series(diag)
    iterations = get_iterations(ts, stride=stride)

    if not iterations:
        raise RuntimeError(f"No iterations found in diagnostic: {diag}")

    rows: list[dict[str, Any]] = []

    for it in iterations:
        print(f"[READ] iteration {it}")

        Ex = read_field_rz(ts, it, field="E", coord="x", positive_r=True)
        Ey = read_field_rz(ts, it, field="E", coord="y", positive_r=True)
        Ez = read_field_rz(ts, it, field="E", coord="z", positive_r=True)

        laser, _ = weighted_laser_metrics(
            Ex.arr_rz,
            Ey.arr_rz,
            Ex.r_um,
            Ex.z_um,
            smooth_um=smooth_um,
            lambda0_m=lambda0_m,
        )

        wake = wake_metrics(
            Ez.arr_rz,
            Ez.r_um,
            Ez.z_um,
            laser["z_peak_um"],
            wake_behind_um=wake_behind_um,
            wake_gap_um=wake_gap_um,
        )

        time_s = getattr(Ex.info, "time", np.nan)

        row = {
            "iteration": int(it),
            "time_fs": float(time_s) * 1.0e15 if np.isfinite(time_s) else float("nan"),
            "z_min_um": float(Ex.z_um.min()),
            "z_max_um": float(Ex.z_um.max()),
            **laser,
            **wake,
        }
        rows.append(row)

    first_valid_laser_index(rows)
    add_propagation_columns(rows)
    return rows


def write_case_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path
