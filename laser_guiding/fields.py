"""Field representations with explicit modes, units and spatial references."""
from __future__ import annotations

import numpy as np

from .openpmd_io import RZField, read_field_rz

QE = 1.602176634e-19


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values
    left, right = width // 2, width - 1 - width // 2
    padded = np.pad(values, ((0, 0), (left, right)), mode="edge")
    integral = np.concatenate((np.zeros((len(values), 1)),
                               np.cumsum(padded, axis=1)), axis=1)
    return (integral[:, width:] - integral[:, :-width]) / width


def validate_plane(f: RZField) -> None:
    if not np.all(np.isfinite(f.arr_rz)):
        raise ValueError("Non-finite reconstructed field")
    for axis in (f.r_um, f.z_um):
        if len(axis) < 2 or not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
            raise ValueError("Coordinates must be finite, increasing and have at least two cells")
    if not (np.any(f.r_um < 0) and np.any(f.r_um > 0)):
        raise ValueError("Expected the reader's full signed transverse plane")


def read_quantity(ts, iteration: int, spec: dict, theta: float, *,
                  wavelength_m: float | None = None) -> RZField:
    kwargs = dict(m=spec["m"], theta=theta)
    if spec["kind"] == "transverse_e2":
        a = read_field_rz(ts, iteration, "E", "x", **kwargs)
        b = read_field_rz(ts, iteration, "E", "y", **kwargs)
        if not (np.array_equal(a.r_um, b.r_um) and np.array_equal(a.z_um, b.z_um)):
            raise ValueError("Transverse component coordinates differ")
        data = a.arr_rz**2 + b.arr_rz**2
        smooth = float(spec.get("smooth_wavelengths", 0))
        if smooth:
            if wavelength_m is None or wavelength_m <= 0:
                raise ValueError("Smoothing requires an explicit positive wavelength")
            spacing = np.diff(a.z_um)
            if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-12):
                raise ValueError("Smoothing requires a uniform longitudinal mesh")
            width = max(1, int(round(smooth * wavelength_m * 1e6 / spacing[0])))
            data = rolling_mean(data, width)
    else:
        a = read_field_rz(ts, iteration, spec["field"], spec.get("coord"), **kwargs)
        data = a.arr_rz
        if spec["kind"] == "electron_density":
            data = -data / QE
    result = RZField(data * float(spec.get("unit_scale", 1)), a.r_um, a.z_um, a.info)
    validate_plane(result)
    return result


def reference_z_m(reference: dict, time_s: float, plane: RZField) -> float:
    if reference["kind"] == "absolute":
        return 0.0
    if reference["kind"] == "linear":
        return float(reference["initial_z_m"]) + float(reference["velocity_m_s"]) * time_s
    if reference["kind"] == "window_center":
        return float((plane.z_um[0] + plane.z_um[-1]) * 0.5e-6)
    raise ValueError("Unknown reference")


def crop(plane: RZField, view: dict, z_reference_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = plane.r_um
    z = plane.z_um - z_reference_m * 1e6
    ri = (r >= view["transverse_um"][0]) & (r <= view["transverse_um"][1])
    zi = (z >= view["longitudinal_um"][0]) & (z <= view["longitudinal_um"][1])
    if np.count_nonzero(ri) < 2 or np.count_nonzero(zi) < 2:
        raise ValueError(f"View {view['name']} contains fewer than two cells on an axis")
    return plane.arr_rz[np.ix_(ri, zi)], r[ri], z[zi]
