from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries


EXPECTED_RZ_AXES = {0: "z", 1: "r"}


@dataclass(frozen=True)
class RZField:
    """WarpX RZ field oriented as arr[r, z], with coordinates in microns."""

    arr_rz: np.ndarray
    r_um: np.ndarray
    z_um: np.ndarray
    info: Any


def open_series(diag: str | Path) -> OpenPMDTimeSeries:
    diag = Path(diag)
    if not diag.exists():
        raise FileNotFoundError(f"Diagnostic directory does not exist: {diag}")
    return OpenPMDTimeSeries(str(diag))


def get_iterations(ts: OpenPMDTimeSeries, stride: int = 1) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return [int(it) for it in ts.iterations[::stride]]


def _as_2d(data: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(data))
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D RZ field array after squeeze, got shape={arr.shape}"
        )
    return arr


def rz_signed(data: np.ndarray, info: Any) -> RZField:
    """Return signed-r RZ data as arr[r, z].

    openPMD-viewer returns the WarpX RZ mesh as data[z, r] with
    info.axes == {0: 'z', 1: 'r'} for the files we are analyzing.
    """
    arr_zr = _as_2d(data)

    axes = getattr(info, "axes", None)
    if axes != EXPECTED_RZ_AXES:
        raise ValueError(f"Unexpected axes={axes}. Expected {EXPECTED_RZ_AXES}")

    z_um = np.asarray(info.z) * 1.0e6
    r_um = np.asarray(info.r) * 1.0e6

    if arr_zr.shape != (len(z_um), len(r_um)):
        raise ValueError(
            "Field shape is inconsistent with RZ coordinates: "
            f"shape={arr_zr.shape}, len(z)={len(z_um)}, len(r)={len(r_um)}"
        )

    return RZField(arr_rz=arr_zr.T, r_um=r_um, z_um=z_um, info=info)


def rz_positive(data: np.ndarray, info: Any) -> RZField:
    """Return r >= 0 part of an RZ field as arr[r>=0, z]."""
    f = rz_signed(data, info)
    mask = f.r_um >= 0.0
    return RZField(
        arr_rz=f.arr_rz[mask, :], r_um=f.r_um[mask], z_um=f.z_um, info=f.info
    )


def read_field_rz(
    ts: OpenPMDTimeSeries,
    iteration: int,
    field: str,
    coord: str | None = None,
    positive_r: bool = False,
) -> RZField:
    """Read a field/component from openPMD-viewer and orient it as arr[r, z]."""
    if coord is None:
        data, info = ts.get_field(iteration=iteration, field=field)
    else:
        data, info = ts.get_field(iteration=iteration, field=field, coord=coord)

    if positive_r:
        return rz_positive(data, info)
    return rz_signed(data, info)


def describe_series(ts: OpenPMDTimeSeries) -> dict[str, Any]:
    """Small serializable summary useful for logs/debugging."""
    return {
        "iterations": [int(it) for it in ts.iterations],
        "avail_fields": list(getattr(ts, "avail_fields", [])),
        "avail_species": list(getattr(ts, "avail_species", [])),
    }
