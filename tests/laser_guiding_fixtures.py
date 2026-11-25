"""Small real HDF5/openPMD fixtures; no WarpX execution or mocked data reader."""
from pathlib import Path

import h5py
import numpy as np

QE = 1.602176634e-19
ME = 9.1093837015e-31
C = 299792458.0


def _root(path, iteration, time_s):
    f = h5py.File(path, "w")
    for name, value in {
        "openPMD": "1.1.0", "basePath": "/data/%T/",
        "iterationEncoding": "fileBased", "iterationFormat": "data%T.h5",
        "meshesPath": "fields/", "particlesPath": "particles/",
    }.items():
        f.attrs[name] = np.bytes_(value)
    f.attrs["openPMDextension"] = np.uint32(0)
    it = f.create_group(f"data/{iteration}")
    it.attrs.update(time=time_s, timeUnitSI=1.0, dt=1e-15)
    return f, it


def _mesh_attrs(obj, zlo, order):
    obj.attrs["geometry"] = np.bytes_("thetaMode")
    obj.attrs["dataOrder"] = np.bytes_("C")
    obj.attrs["axisLabels"] = np.asarray([np.bytes_(a) for a in order])
    mapping = {"z": (2e-6, zlo), "r": (1e-6, 0.0)}
    obj.attrs["gridSpacing"] = np.asarray([mapping[a][0] for a in order])
    obj.attrs["gridGlobalOffset"] = np.asarray([mapping[a][1] for a in order])
    obj.attrs["gridUnitSI"] = 1.0
    obj.attrs["timeOffset"] = 0.0
    obj.attrs["unitDimension"] = np.zeros(7)


def _component(parent, name, data):
    ds = parent.create_dataset(name, data=data)
    ds.attrs["unitSI"] = 1.0
    ds.attrs["position"] = np.asarray([0.5, 0.5])
    return ds


def fields_file(path, iteration=10, time_s=1e-13, order=("z", "r"), zref=0.0):
    f, it = _root(path, iteration, time_s)
    with f:
        fields = it.create_group("fields")
        shape = (3, 6, 3) if order == ("z", "r") else (3, 3, 6)
        rho = np.zeros(shape)
        rho[0] = -QE * 1e22
        rho[1] = -QE * 2e21
        rho[2] = -QE * 3e21
        ds = _component(fields, "rho_background", rho)
        _mesh_attrs(ds, zref - 6e-6, order)
        electric = fields.create_group("E")
        _mesh_attrs(electric, zref - 6e-6, order)
        for coord in ("r", "t", "z"):
            data = np.zeros(shape)
            if coord == "r":
                data[1] = 2e12
            elif coord == "t":
                data[2] = -2e12
            else:
                data[0] = 4e9
                data[1] = 7e9
            _component(electric, coord, data)


def _constant(parent, name, value, n):
    group = parent.create_group(name)
    group.attrs.update(value=value, shape=np.asarray([n], dtype=np.uint64), unitSI=1.0)


def particles_file(path, iteration=10, time_s=1e-13, species="background", empty=False,
                   zref=0.0, weights=None):
    f, it = _root(path, iteration, time_s)
    with f:
        sp = it.create_group("particles").create_group(species)
        w = np.asarray([] if empty else ([2e7, 5e7, 7e7] if weights is None else weights))
        n = w.size
        ds = sp.create_dataset("weighting", data=w)
        ds.attrs["unitSI"] = 1.0
        position, offset, momentum = [sp.create_group(name) for name in
                                      ("position", "positionOffset", "momentum")]
        for coord in ("x", "y", "z"):
            ds = position.create_dataset(coord, data=np.full(n, zref if coord == "z" else 0.0))
            ds.attrs["unitSI"] = 1.0
            _constant(offset, coord, 0.0, n)
            u = np.asarray([] if empty else ([0.0, 1.0, -3.0] if coord == "z" else [0.0] * n))
            ds = momentum.create_dataset(coord, data=u * ME * C)
            ds.attrs["unitSI"] = 1.0
        _constant(sp, "mass", ME, n)
        _constant(sp, "charge", -QE, n)


def make_case(root, *, particle=True):
    case = Path(root) / "case"
    (case / "diags" / "fields").mkdir(parents=True)
    if particle:
        (case / "diags" / "plasma_electrons").mkdir()
    for iteration, t in ((10, 1e-13), (20, 2e-13)):
        zref = -333e-6 + C * t
        fields_file(case / "diags" / "fields" / f"fields_{iteration}.h5",
                    iteration, t, zref=zref)
        if particle:
            particles_file(case / "diags" / "plasma_electrons" / f"particles_{iteration}.h5",
                           iteration, t, zref=zref)
    return case


def config():
    return {
        "schema_version": 1, "field_diag": "diags/fields",
        "reference": {"kind": "linear", "initial_z_m": -333e-6, "velocity_m_s": C},
        "wavelength_m": 1.06e-6,
        "views": [{"name": "xz", "theta_rad": 0.0,
                   "transverse_um": [-4, 4], "longitudinal_um": [-7, 7]}],
        "fields": [{"name": "density", "kind": "electron_density", "field": "rho_background",
                    "m": "all", "unit_scale": 1e-6, "unit": "cm^-3", "norm": "log"}],
        "particles": {"diag": "diags/plasma_electrons", "species": ["background"],
                      "energy_bins": {"min": 0, "max": 10, "count": 20},
                      "longitudinal_bins": {"min": -7, "max": 7, "count": 20}},
    }
