#!/usr/bin/env python3
"""Write small WarpX-like thetaMode openPMD series for the C++ parity tests.

The fields are analytic (no random numbers), so regenerating the fixtures gives
the same values. tools/golden/generate_golden.py then runs the Python reference
pipeline on them.

Coverage built into the three cases:
- channel: E/t on a nodal radial grid (mirrored r = -0.0 row), E/r and E/z with
  different staggering, contiguous float64 datasets
- uniform: cell-centred everywhere, non-zero timeOffset
- vacuum: float32, gzip-chunked datasets (HDF5 read path), E/z unitSI != 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "cpp" / "tests" / "data" / "synthetic" / "campaign"

N_MODES = 3  # m = 0 plus the cos/sin parts of m = 1 (WarpX n_rz_azimuthal_modes = 2)
N_Z = 160
N_R = 16
DZ = 0.8e-6
DR = 2.0e-6
DT = 1.0e-15
ITERATIONS = (0, 5400, 6100, 6800)
WINDOW_OFFSET_UM = (0.0, 5400.0, 6100.0, 6800.0)  # moving-window z offset per iteration
E0 = 4.0e12
WAKE_AMPLITUDE = 5.0e10
CARRIER_WAVELENGTH = 6.4e-6
PULSE_LENGTH = 6.0e-6
PLASMA_WAVELENGTH = 30.0e-6


@dataclass(frozen=True)
class Case:
    name: str
    waist_um: tuple[float, ...]
    positions: dict[str, tuple[float, float]]  # component -> (z, r) staggering
    dtype: str = "f8"
    chunked: bool = False
    time_offset: float = 0.0
    ez_unit_si: float = 1.0
    ey_fraction: float = 0.05
    extra: dict[str, float] = field(default_factory=dict)


CASES = (
    Case(
        name="000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz",
        waist_um=(10.0, 10.4, 9.8, 10.2),
        positions={"r": (0.0, 0.5), "t": (0.0, 0.0), "z": (0.5, 0.0)},
    ),
    Case(
        name="001_f20_uni_n4e18cm3_L2mm_refd150um_foc0um_rz",
        waist_um=(10.0, 11.5, 12.4, 13.1),
        positions={"r": (0.5, 0.5), "t": (0.5, 0.5), "z": (0.5, 0.5)},
        time_offset=2.5e-16,
    ),
    Case(
        name="002_f20_vac_L2mm_refd150um_foc0um_rz",
        waist_um=(10.0, 14.0, 16.5, 18.2),
        positions={"r": (0.0, 0.0), "t": (0.0, 0.0), "z": (0.0, 0.0)},
        dtype="f4",
        chunked=True,
        ez_unit_si=1.0e3,
        ey_fraction=0.0,
    ),
)


def component_modes(case: Case, component: str, k: int) -> np.ndarray:
    pz, pr = case.positions[component]
    z0 = WINDOW_OFFSET_UM[k] * 1.0e-6
    z = (z0 + (np.arange(N_Z) + pz) * DZ)[:, None]
    r = ((np.arange(N_R) + pr) * DR)[None, :]
    zc = z0 + 0.84 * (N_Z - 1) * DZ
    waist = case.waist_um[k] * 1.0e-6
    amplitude = E0 * case.waist_um[0] / case.waist_um[k]

    envelope = np.exp(-(((z - zc) / PULSE_LENGTH) ** 2))
    radial = np.exp(-((r / waist) ** 2))
    carrier = 2.0 * np.pi * (z - zc) / CARRIER_WAVELENGTH
    laser_cos = radial * envelope * np.cos(carrier)
    laser_sin = radial * envelope * np.sin(carrier)

    modes = np.zeros((N_MODES, N_Z, N_R))
    if component == "r":
        modes[0] = 0.03 * amplitude * (r / waist) * radial * envelope**2
        modes[1] = amplitude * laser_cos
        modes[2] = 0.25 * amplitude * laser_sin
    elif component == "t":
        modes[1] = case.ey_fraction * amplitude * laser_cos
        modes[2] = -amplitude * laser_cos
    elif component == "z":
        behind = np.clip(zc - z, 0.0, None)
        wake = (
            WAKE_AMPLITUDE
            * np.sin(2.0 * np.pi * behind / PLASMA_WAVELENGTH)
            * np.exp(-behind / 80.0e-6)
            * (z < zc)
            * np.exp(-((r / (1.5 * waist)) ** 2))
        )
        modes[0] = wake
        modes[1] = 0.2 * wake * (r / waist)
        modes[2] = 0.1 * wake * (r / waist)
        modes /= case.ez_unit_si
    return modes.astype(case.dtype)


def write_iteration(path: Path, case: Case, k: int) -> None:
    iteration = ITERATIONS[k]
    with h5py.File(path, "w") as f:
        f.attrs["openPMD"] = np.bytes_("1.1.0")
        f.attrs["openPMDextension"] = np.uint32(1)
        f.attrs["basePath"] = np.bytes_("/data/%T/")
        f.attrs["meshesPath"] = np.bytes_("fields/")
        f.attrs["particlesPath"] = np.bytes_("particles/")
        f.attrs["iterationEncoding"] = np.bytes_("fileBased")
        f.attrs["iterationFormat"] = np.bytes_("openpmd_%06T.h5")
        f.attrs["software"] = np.bytes_("guiding_analysis_module synthetic fixture")

        it = f.create_group(f"data/{iteration}")
        it.attrs["time"] = np.float64(iteration * DT)
        it.attrs["dt"] = np.float64(DT)
        it.attrs["timeUnitSI"] = np.float64(1.0)

        record = it.create_group("fields/E")
        record.attrs["geometry"] = np.bytes_("thetaMode")
        record.attrs["geometryParameters"] = np.bytes_("m=1;imag=+")
        record.attrs["dataOrder"] = np.bytes_("C")
        record.attrs["axisLabels"] = np.array([b"z", b"r"])
        record.attrs["gridSpacing"] = np.array([DZ, DR])
        record.attrs["gridGlobalOffset"] = np.array([WINDOW_OFFSET_UM[k] * 1.0e-6, 0.0])
        record.attrs["gridUnitSI"] = np.float64(1.0)
        record.attrs["unitDimension"] = np.array([1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0])
        record.attrs["timeOffset"] = np.float64(case.time_offset)
        record.attrs["fieldSmoothing"] = np.bytes_("none")

        for component in ("r", "t", "z"):
            data = component_modes(case, component, k)
            options = {"chunks": (1, 64, N_R), "compression": "gzip", "compression_opts": 4} if case.chunked else {}
            dataset = record.create_dataset(component, data=data, track_times=False, **options)
            dataset.attrs["position"] = np.array(case.positions[component], dtype=np.float64)
            dataset.attrs["unitSI"] = np.float64(case.ez_unit_si if component == "z" else 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    campaign = args.outdir
    rows = ["case_id\tcase_dir"]
    for case in CASES:
        diag = campaign / case.name / "diags" / "diag1"
        diag.mkdir(parents=True, exist_ok=True)
        for old in diag.glob("*.h5"):
            old.unlink()
        for k, iteration in enumerate(ITERATIONS):
            write_iteration(diag / f"openpmd_{iteration:06d}.h5", case, k)
        rows.append(f"{case.name}\t{case.name}")
    (campaign / "cases_full.tsv").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(CASES)} cases x {len(ITERATIONS)} iterations to {campaign}")


if __name__ == "__main__":
    main()
