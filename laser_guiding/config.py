"""Explicit, JSON-serializable conventions for a visualization run."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _only(obj: dict, allowed: set[str], name: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise ValueError(f"Unknown {name} keys: {sorted(extra)}")


def _range(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [lower, upper]")
    if number(value[0], name) >= number(value[1], name):
        raise ValueError(f"{name} must have lower < upper")


def _resolve(value: Any, parameters: dict) -> Any:
    if isinstance(value, dict):
        if set(value) == {"parameter"}:
            result = parameters
            for key in value["parameter"].split("."):
                result = result[key]
            return copy.deepcopy(result)
        return {key: _resolve(item, parameters) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, parameters) for item in value]
    return value


def load_config(path: str | Path, case_dir: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = {}
    if raw.get("parameter_file"):
        parameters = json.loads((Path(case_dir) / raw["parameter_file"]).read_text())
        if parameters.get("dry_run"):
            raise ValueError("Parameter file describes a preflight-only case")
    config = _resolve(raw, parameters)
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    _only(c, {"schema_version", "field_diag", "parameter_file", "completed_log",
              "reference", "wavelength_m", "views", "fields", "particles"}, "config")
    if c.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if not isinstance(c.get("field_diag"), str) or not c["field_diag"]:
        raise ValueError("field_diag must explicitly name a diagnostic directory")
    ref = c["reference"]
    _only(ref, {"kind", "initial_z_m", "velocity_m_s"}, "reference")
    if ref["kind"] not in {"absolute", "linear", "window_center"}:
        raise ValueError("reference.kind must be absolute, linear or window_center")
    if ref["kind"] == "linear":
        number(ref["initial_z_m"], "reference.initial_z_m")
        number(ref["velocity_m_s"], "reference.velocity_m_s")
    elif set(ref) != {"kind"}:
        raise ValueError("Only a linear reference accepts initial_z_m/velocity_m_s")
    if "wavelength_m" in c and number(c["wavelength_m"], "wavelength_m") <= 0:
        raise ValueError("wavelength_m must be positive")
    for collection in ("views", "fields"):
        items = c[collection]
        if not items or len({x["name"] for x in items}) != len(items):
            raise ValueError(f"{collection} must be nonempty with unique names")
        for item in items:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item["name"]):
                raise ValueError("Names may contain only letters, digits, _ and -")
    for v in c["views"]:
        _only(v, {"name", "theta_rad", "transverse_um", "longitudinal_um"}, "view")
        number(v["theta_rad"], "theta_rad")
        _range(v["transverse_um"], "transverse_um")
        _range(v["longitudinal_um"], "longitudinal_um")
    for f in c["fields"]:
        _only(f, {"name", "kind", "field", "coord", "m", "unit_scale", "unit",
                  "label", "norm", "vmin", "vmax", "smooth_wavelengths"}, "field")
        if f["kind"] not in {"field", "electron_density", "transverse_e2"}:
            raise ValueError("Unknown field kind")
        m = f["m"]
        if m != "all" and (type(m) is not int or m < 0):
            raise ValueError("m must be 'all' or a nonnegative integer")
        if f.get("norm", "linear") not in {"linear", "symmetric", "log"}:
            raise ValueError("norm must be linear, symmetric or log")
        if number(f.get("unit_scale", 1.0), "unit_scale") <= 0:
            raise ValueError("unit_scale must be positive")
        if f["kind"] != "transverse_e2" and not f.get("field"):
            raise ValueError("field name is required")
        if f["kind"] == "electron_density" and (not f["field"].startswith("rho_") or f.get("coord") is not None):
            raise ValueError("electron_density requires a species charge-density rho_* scalar")
        smooth = number(f.get("smooth_wavelengths", 0.0), "smooth_wavelengths")
        if smooth < 0 or (smooth > 0 and (f["kind"] != "transverse_e2" or "wavelength_m" not in c)):
            raise ValueError("Smoothing requires transverse_e2 and an explicit wavelength_m")
        for key in ("vmin", "vmax"):
            if key in f:
                number(f[key], key)
        if "vmin" in f and "vmax" in f and f["vmin"] >= f["vmax"]:
            raise ValueError("vmin must be below vmax")
        if f.get("norm") == "log" and any(f.get(k, 1) <= 0 for k in ("vmin", "vmax")):
            raise ValueError("Logarithmic limits must be positive")
        if f.get("norm") == "symmetric" and ("vmin" in f) != ("vmax" in f):
            raise ValueError("Supply both symmetric limits or neither")
        if f.get("norm") == "symmetric" and "vmin" in f and f["vmin"] != -f["vmax"]:
            raise ValueError("Symmetric limits must satisfy vmin=-vmax")
    if c.get("particles") is not None:
        p = c["particles"]
        _only(p, {"diag", "species", "forward_only", "energy_min_mev",
                  "energy_bins", "longitudinal_bins"}, "particles")
        if not p.get("diag") or not p.get("species") or len(set(p["species"])) != len(p["species"]):
            raise ValueError("Specify a particle diagnostic and unique electron species")
        if not isinstance(p.get("forward_only", False), bool):
            raise ValueError("forward_only must be boolean")
        if number(p.get("energy_min_mev", 0.0), "energy_min_mev") < 0:
            raise ValueError("energy_min_mev must be nonnegative")
        bin_edges(p["energy_bins"])
        bin_edges(p["longitudinal_bins"])


def bin_edges(spec: dict) -> np.ndarray:
    _only(spec, {"min", "max", "count", "spacing"}, "bins")
    lo, hi = number(spec["min"], "bin min"), number(spec["max"], "bin max")
    count = spec["count"]
    if type(count) is not int or count < 1 or lo >= hi:
        raise ValueError("Bins require count >= 1 and min < max")
    if spec.get("spacing", "linear") == "log":
        if lo <= 0:
            raise ValueError("Logarithmic bins require min > 0")
        return np.geomspace(lo, hi, count + 1)
    if spec.get("spacing", "linear") != "linear":
        raise ValueError("Unknown bin spacing")
    return np.linspace(lo, hi, count + 1)
