"""Charge histograms of electron populations currently inside the saved window."""
from __future__ import annotations

import numpy as np

QE = 1.602176634e-19
ME_C2_MEV = 0.51099895069


def kinetic_energy_mev(ux, uy, uz):
    # Algebraically equivalent to sqrt(1+u^2)-1, without cold-electron cancellation.
    u2 = ux * ux + uy * uy + uz * uz
    return ME_C2_MEV * u2 / (np.sqrt(1 + u2) + 1)


def histograms(ts, iteration: int, spec: dict, z_reference_m: float,
               energy_edges: np.ndarray, longitudinal_edges: np.ndarray, *, transverse=None):
    if iteration not in set(map(int, ts.iterations)):
        raise ValueError(f"Exact particle iteration {iteration} is absent")
    spectrum = np.zeros(len(energy_edges) - 1)
    phase = np.zeros((len(longitudinal_edges) - 1, len(energy_edges) - 1))
    if transverse is not None:
        position_edges = transverse["position_edges_um"]
        momentum_edges = transverse["momentum_edges_mc"]
    rows = []
    for species in spec["species"]:
        if species not in (ts.avail_species or []):
            raise ValueError(f"Requested electron species is absent: {species}")
        w, = ts.get_particle(var_list=["w"], species=species, iteration=iteration)
        w = np.asarray(w)
        if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(f"Invalid electron weights: {species}")
        row = dict(species=species, registered=True, macroparticles=int(w.size),
                   selected_macroparticles=0, Q_window_pC=float(w.sum() * QE * 1e12),
                   Q_selected_pC=0.0, Q_ge_10_MeV_pC=0.0, Q_energy_underflow_pC=0.0,
                   Q_energy_overflow_pC=0.0, Q_phase_outside_pC=0.0)
        if transverse is not None:
            row.update(Q_x_ux_outside_pC=0.0, Q_y_uy_outside_pC=0.0)
        if w.size:
            names = ["z", "ux", "uy", "uz"] + (["x", "y"] if transverse is not None else [])
            arrays = [np.asarray(a) for a in ts.get_particle(
                var_list=names, species=species, iteration=iteration)]
            z, ux, uy, uz = arrays[:4]
            if any(a.shape != w.shape or not np.all(np.isfinite(a)) for a in arrays):
                raise ValueError(f"Invalid electron arrays: {species}")
            energy = kinetic_energy_mev(ux, uy, uz)
            row["Q_ge_10_MeV_pC"] = float(w[energy >= 10.0].sum() * QE * 1e12)
            valid = (w > 0) & (energy >= spec.get("energy_min_mev", 0.0))
            if spec.get("forward_only", False):
                valid &= uz > 0
            energy, charge = energy[valid], w[valid] * QE * 1e12
            xi = (z[valid] - z_reference_m) * 1e6
            spectrum += np.histogram(energy, bins=energy_edges, weights=charge)[0]
            phase += np.histogram2d(xi, energy, bins=(longitudinal_edges, energy_edges),
                                    weights=charge)[0]
            outside = ((xi < longitudinal_edges[0]) | (xi > longitudinal_edges[-1]) |
                       (energy < energy_edges[0]) | (energy > energy_edges[-1]))
            row.update(selected_macroparticles=int(energy.size), Q_selected_pC=float(charge.sum()),
                       Q_energy_underflow_pC=float(charge[energy < energy_edges[0]].sum()),
                       Q_energy_overflow_pC=float(charge[energy > energy_edges[-1]].sum()),
                       Q_phase_outside_pC=float(charge[outside].sum()))
            if transverse is not None:
                for axis, position, momentum in (("x", arrays[4], ux), ("y", arrays[5], uy)):
                    position_um = position[valid] * 1e6
                    u = momentum[valid]
                    histogram = np.histogram2d(position_um, u,
                                               bins=(position_edges, momentum_edges),
                                               weights=charge)[0]
                    transverse[f"charge_{axis}_u{axis}_pc"] += histogram
                    outside_transverse = ((position_um < position_edges[0]) |
                                          (position_um > position_edges[-1]) |
                                          (u < momentum_edges[0]) | (u > momentum_edges[-1]))
                    row[f"Q_{axis}_u{axis}_outside_pC"] = float(charge[outside_transverse].sum())
        rows.append(row)
    return spectrum, phase, rows
