"""RZ field access without assumptions about a guiding or injection scheme."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

EXPECTED_RZ_AXES = {0: "z", 1: "r"}


@dataclass(frozen=True)
class RZField:
    """A reconstructed angular plane, arr[r, z], with coordinates in micrometres."""

    arr_rz: np.ndarray
    r_um: np.ndarray
    z_um: np.ndarray
    info: Any


def open_series(diag: str | Path, *, backend: str | None = None,
                check_all_files: bool = True) -> OpenPMDTimeSeries:
    diag = Path(diag)
    if not diag.is_dir():
        raise FileNotFoundError(f"Diagnostic directory does not exist: {diag}")
    return OpenPMDTimeSeries(str(diag), backend=backend, check_all_files=check_all_files)


def get_iterations(ts: OpenPMDTimeSeries, stride: int = 1) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return [int(it) for it in ts.iterations[::stride]]


def _as_2d(data: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(data))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D RZ field array after squeeze, got shape={arr.shape}")
    return arr


def rz_signed(data: np.ndarray, info: Any) -> RZField:
    """Keep both halves reconstructed by the reader; never mirror a positive cut."""
    arr = _as_2d(data)
    axes = getattr(info, "axes", None)
    if axes == {0: "z", 1: "r"}:
        arr = arr.T
    elif axes != {0: "r", 1: "z"}:
        raise ValueError(f"Unexpected axes={axes}; expected the two RZ axes")
    r_um = np.asarray(info.r) * 1e6
    z_um = np.asarray(info.z) * 1e6
    if arr.shape != (len(r_um), len(z_um)):
        raise ValueError("Field shape is inconsistent with RZ coordinates")
    return RZField(arr, r_um, z_um, info)


def rz_positive(data: np.ndarray, info: Any) -> RZField:
    f = rz_signed(data, info)
    mask = f.r_um >= 0
    return RZField(f.arr_rz[mask], f.r_um[mask], f.z_um, f.info)


def read_field_rz(ts: OpenPMDTimeSeries, iteration: int, field: str,
                  coord: str | None = None, positive_r: bool = False, *,
                  m: int | str = "all", theta: float = 0.0) -> RZField:
    if int(iteration) not in set(map(int, ts.iterations)):
        raise ValueError(f"Exact field iteration {iteration} is absent")
    kwargs = dict(iteration=int(iteration), field=field, m=m, theta=theta)
    if coord is not None:
        kwargs["coord"] = coord
    data, info = ts.get_field(**kwargs)
    return (rz_positive if positive_r else rz_signed)(data, info)


def describe_series(ts: OpenPMDTimeSeries) -> dict[str, Any]:
    return {
        "iterations": [int(it) for it in ts.iterations],
        "avail_fields": list(getattr(ts, "avail_fields", None) or []),
        "avail_species": list(getattr(ts, "avail_species", None) or []),
    }
