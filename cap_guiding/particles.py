from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import E_CHARGE_C
from .openpmd_io import open_series, get_iterations

ELECTRON_REST_ENERGY_MEV = 0.51099895
DEFAULT_PARTICLE_VARS = ["x", "y", "z", "ux", "uy", "uz", "w"]


@dataclass(frozen=True)
class ParticleDump:
    """Particle arrays for one openPMD iteration.

    Positions are stored in meters by WarpX/openPMD. Momenta are normalized
    momenta ux, uy, uz = gamma * v/c. Weight w is the macroparticle weight.
    """

    iteration: int
    time_fs: float
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    w: np.ndarray

    @property
    def gamma(self) -> np.ndarray:
        return gamma_from_u(self.ux, self.uy, self.uz)

    @property
    def kinetic_energy_mev(self) -> np.ndarray:
        return kinetic_energy_mev_from_u(self.ux, self.uy, self.uz)


def gamma_from_u(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    return np.sqrt(1.0 + ux * ux + uy * uy + uz * uz)


def kinetic_energy_mev_from_u(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> np.ndarray:
    return (gamma_from_u(ux, uy, uz) - 1.0) * ELECTRON_REST_ENERGY_MEV


def read_particle_dump(
    diag: str | Path,
    *,
    species: str,
    iteration: int,
) -> ParticleDump:
    """Read one particle openPMD dump written by WarpX ParticleDiagnostic."""
    ts = open_series(diag)
    arrays = ts.get_particle(
        var_list=DEFAULT_PARTICLE_VARS,
        species=species,
        iteration=int(iteration),
    )

    if len(arrays) != len(DEFAULT_PARTICLE_VARS):
        raise RuntimeError(
            "Unexpected number of arrays returned by openPMD-viewer: "
            f"got {len(arrays)}, expected {len(DEFAULT_PARTICLE_VARS)}"
        )

    x, y, z, ux, uy, uz, w = [np.asarray(a) for a in arrays]
    n = len(w)

    for name, arr in zip(DEFAULT_PARTICLE_VARS, [x, y, z, ux, uy, uz, w]):
        if len(arr) != n:
            raise ValueError(
                f"Particle array length mismatch for {name}: len={len(arr)}, len(w)={n}"
            )

    time_s = getattr(ts, "t", None)
    time_fs = float("nan")
    if time_s is not None:
        try:
            idx = list(map(int, ts.iterations)).index(int(iteration))
            time_fs = float(np.asarray(time_s)[idx]) * 1.0e15
        except Exception:
            time_fs = float("nan")

    return ParticleDump(
        iteration=int(iteration),
        time_fs=time_fs,
        x_m=x,
        y_m=y,
        z_m=z,
        ux=ux,
        uy=uy,
        uz=uz,
        w=w,
    )


def last_iteration(diag: str | Path) -> int:
    ts = open_series(diag)
    iterations = get_iterations(ts, stride=1)
    if not iterations:
        raise RuntimeError(f"No iterations found in particle diagnostic: {diag}")
    return int(iterations[-1])


def select_hot_electrons(
    dump: ParticleDump,
    *,
    hot_energy_mev: float = 10.0,
    forward_only: bool = True,
    longitudinal: str = "z",
    exit_window_mm: float | None = None,
) -> np.ndarray:
    """Return a boolean mask selecting accelerated electrons."""
    energy = dump.kinetic_energy_mev
    mask = np.isfinite(energy) & np.isfinite(dump.w) & (dump.w > 0.0)
    mask &= energy >= float(hot_energy_mev)

    if longitudinal == "z":
        q_long = dump.z_m
        u_long = dump.uz
    elif longitudinal == "x":
        q_long = dump.x_m
        u_long = dump.ux
    elif longitudinal == "y":
        q_long = dump.y_m
        u_long = dump.uy
    else:
        raise ValueError("longitudinal must be one of: x, y, z")

    mask &= np.isfinite(q_long) & np.isfinite(u_long)

    if forward_only:
        mask &= u_long > 0.0

    if exit_window_mm is not None:
        window_m = float(exit_window_mm) * 1.0e-3
        if window_m <= 0.0:
            raise ValueError("exit_window_mm must be positive when provided")

        qmax = float(np.nanmax(q_long[mask])) if np.any(mask) else float("nan")
        if np.isfinite(qmax):
            mask &= q_long >= qmax - window_m

    return mask


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")

    values = values[finite]
    weights = weights[finite]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    target = float(percentile) / 100.0 * cumulative[-1]

    return float(values[np.searchsorted(cumulative, target, side="left")])


def summarize_dump(
    dump: ParticleDump,
    *,
    hot_energy_mev: float = 10.0,
    longitudinal: str = "z",
    exit_window_mm: float | None = None,
    forward_only: bool = True,
) -> dict[str, Any]:
    energy = dump.kinetic_energy_mev
    w = dump.w

    finite = np.isfinite(energy) & np.isfinite(w) & (w > 0.0)

    hot = select_hot_electrons(
        dump,
        hot_energy_mev=hot_energy_mev,
        forward_only=forward_only,
        longitudinal=longitudinal,
        exit_window_mm=exit_window_mm,
    )

    if longitudinal == "z":
        q_long = dump.z_m
        u_long = dump.uz
    elif longitudinal == "x":
        q_long = dump.x_m
        u_long = dump.ux
    else:
        q_long = dump.y_m
        u_long = dump.uy

    hot_weight = float(np.sum(w[hot])) if np.any(hot) else 0.0
    total_weight = float(np.sum(w[finite])) if np.any(finite) else 0.0

    row: dict[str, Any] = {
        "iteration": int(dump.iteration),
        "time_fs": float(dump.time_fs),
        "n_macroparticles_total": int(len(w)),
        "n_macroparticles_valid": int(np.count_nonzero(finite)),
        "n_macroparticles_hot": int(np.count_nonzero(hot)),
        "weight_total": total_weight,
        "weight_hot": hot_weight,
        "charge_hot_pC": hot_weight * E_CHARGE_C / 1.0e-12,
        "hot_energy_threshold_MeV": float(hot_energy_mev),
        "forward_only": bool(forward_only),
        "longitudinal_coordinate": longitudinal,
        "exit_window_mm": float(exit_window_mm) if exit_window_mm is not None else "",
        "Emax_MeV": float(np.nanmax(energy[finite]))
        if np.any(finite)
        else float("nan"),
        "E99_MeV": weighted_percentile(energy[finite], w[finite], 99.0)
        if np.any(finite)
        else float("nan"),
        "E95_MeV": weighted_percentile(energy[finite], w[finite], 95.0)
        if np.any(finite)
        else float("nan"),
        "E90_MeV": weighted_percentile(energy[finite], w[finite], 90.0)
        if np.any(finite)
        else float("nan"),
    }

    if np.any(hot):
        row.update(
            {
                "Emax_hot_MeV": float(np.nanmax(energy[hot])),
                "Emean_hot_MeV": float(np.average(energy[hot], weights=w[hot])),
                "E99_hot_MeV": weighted_percentile(energy[hot], w[hot], 99.0),
                "E95_hot_MeV": weighted_percentile(energy[hot], w[hot], 95.0),
                "q_long_mean_hot_mm": float(
                    np.average(q_long[hot], weights=w[hot]) * 1.0e3
                ),
                "q_long_min_hot_mm": float(np.nanmin(q_long[hot]) * 1.0e3),
                "q_long_max_hot_mm": float(np.nanmax(q_long[hot]) * 1.0e3),
                "u_long_mean_hot": float(np.average(u_long[hot], weights=w[hot])),
                "u_long_max_hot": float(np.nanmax(u_long[hot])),
            }
        )
    else:
        row.update(
            {
                "Emax_hot_MeV": float("nan"),
                "Emean_hot_MeV": float("nan"),
                "E99_hot_MeV": float("nan"),
                "E95_hot_MeV": float("nan"),
                "q_long_mean_hot_mm": float("nan"),
                "q_long_min_hot_mm": float("nan"),
                "q_long_max_hot_mm": float("nan"),
                "u_long_mean_hot": float("nan"),
                "u_long_max_hot": float("nan"),
            }
        )

    return row


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No particle summary rows to write")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_energy_spectrum(
    dump: ParticleDump,
    *,
    path: str | Path,
    hot_energy_mev: float = 10.0,
    bins: int = 200,
    emax_mev: float | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    energy = dump.kinetic_energy_mev
    finite = np.isfinite(energy) & np.isfinite(dump.w) & (dump.w > 0.0)

    if not np.any(finite):
        raise ValueError("No finite particles for energy spectrum")

    e = energy[finite]
    w = dump.w[finite]

    if emax_mev is None:
        emax_mev = max(float(np.nanpercentile(e, 99.9)), hot_energy_mev * 1.2)

    emax_mev = max(float(emax_mev), hot_energy_mev * 1.2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(e, bins=int(bins), range=(0.0, emax_mev), weights=w, histtype="step")
    ax.axvline(hot_energy_mev, linestyle="--", linewidth=1.0)
    ax.set_xlabel("electron kinetic energy [MeV]")
    ax.set_ylabel("weighted counts [a.u.]")
    ax.set_title(f"Electron energy spectrum, iteration {dump.iteration}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_longitudinal_phase_space(
    dump: ParticleDump,
    *,
    path: str | Path,
    hot_energy_mev: float = 10.0,
    longitudinal: str = "z",
    exit_window_mm: float | None = None,
    forward_only: bool = True,
    max_points: int = 200_000,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hot = select_hot_electrons(
        dump,
        hot_energy_mev=hot_energy_mev,
        forward_only=forward_only,
        longitudinal=longitudinal,
        exit_window_mm=exit_window_mm,
    )

    if longitudinal == "z":
        q = dump.z_m * 1.0e3
        u = dump.uz
        q_label = "z [mm]"
        u_label = "uz"
    elif longitudinal == "x":
        q = dump.x_m * 1.0e3
        u = dump.ux
        q_label = "x [mm]"
        u_label = "ux"
    else:
        q = dump.y_m * 1.0e3
        u = dump.uy
        q_label = "y [mm]"
        u_label = "uy"

    idx = np.where(hot)[0]
    if idx.size == 0:
        raise ValueError("No hot electrons selected for phase-space plot")

    if idx.size > max_points:
        keep = np.linspace(0, idx.size - 1, int(max_points)).astype(int)
        idx = idx[keep]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sc = ax.scatter(
        q[idx],
        u[idx],
        s=1.0,
        c=dump.kinetic_energy_mev[idx],
        alpha=0.35,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Ekin [MeV]")

    ax.set_xlabel(q_label)
    ax.set_ylabel(u_label)
    ax.set_title(f"Hot-electron longitudinal phase space, E > {hot_energy_mev:g} MeV")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path
