"""Compatibility API for existing CLPU consumers.

New scheme-independent readers live in laser_guiding.openpmd_io. Keep the legacy
open_series constructor (including its patch point) and existing units/defaults.
"""
from __future__ import annotations

from pathlib import Path
from openpmd_viewer import OpenPMDTimeSeries

from laser_guiding.openpmd_io import (
    EXPECTED_RZ_AXES, RZField, _as_2d, describe_series, get_iterations,
    rz_positive, rz_signed,
)


def open_series(diag: str | Path) -> OpenPMDTimeSeries:
    diag = Path(diag)
    if not diag.exists():
        raise FileNotFoundError(f"Diagnostic directory does not exist: {diag}")
    return OpenPMDTimeSeries(str(diag))


def read_field_rz(ts: OpenPMDTimeSeries, iteration: int, field: str,
                  coord: str | None = None, positive_r: bool = False) -> RZField:
    """Preserve the legacy call signature and reader defaults for CLPU."""
    if coord is None:
        data, info = ts.get_field(iteration=iteration, field=field)
    else:
        data, info = ts.get_field(iteration=iteration, field=field, coord=coord)
    return (rz_positive if positive_r else rz_signed)(data, info)
