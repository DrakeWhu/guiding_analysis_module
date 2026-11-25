"""Render field and optional electron evolution without CLPU metrics or scores."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from .config import bin_edges, validate_config
from .fields import crop, read_quantity, reference_z_m
from .snapshots import completed_step, inventory, select_iterations, snapshot_series


def _quantities(ts, iteration, config, time_s):
    # Reuse a reconstructed angular plane for all views at that angle.
    for theta in dict.fromkeys(v["theta_rad"] for v in config["views"]):
        for spec in config["fields"]:
            plane = read_quantity(ts, iteration, spec, theta,
                                  wavelength_m=config.get("wavelength_m"))
            if not math.isclose(float(plane.info.time), time_s, rel_tol=1e-10, abs_tol=1e-21):
                raise ValueError(f"Field time offset disagrees with snapshot time at {iteration}")
            zref = reference_z_m(config["reference"], time_s, plane)
            for view in config["views"]:
                if view["theta_rad"] == theta:
                    data, r, z = crop(plane, view, zref)
                    yield spec, view, data, r, z, zref


def _update_scale(stats, data):
    positive = data[data > 0]
    stats["min"] = min(stats["min"], float(data.min()))
    stats["max"] = max(stats["max"], float(data.max()))
    stats["nonpositive_display_samples"] += int(np.count_nonzero(data <= 0))
    if positive.size:
        stats["min_positive"] = min(stats["min_positive"], float(positive.min()))


def _limits(spec, stats):
    kind = spec.get("norm", "linear")
    if kind == "symmetric":
        bound = max(abs(stats["min"]), abs(stats["max"]), 1e-30)
        lo, hi = spec.get("vmin", -bound), spec.get("vmax", bound)
    elif kind == "log":
        lo = spec.get("vmin", stats["min_positive"])
        hi = spec.get("vmax", stats["max"])
        if not math.isfinite(lo) or hi <= 0:
            # An all-zero selected quantity can still be displayed as fully masked.
            if stats["min"] == stats["max"] == 0:
                lo = spec.get("vmin", 1e-30)
                hi = spec.get("vmax", max(lo * 10, 1.0))
            else:
                raise ValueError(f"No positive samples for logarithmic field {spec['name']}")
    else:
        lo, hi = spec.get("vmin", stats["min"]), spec.get("vmax", stats["max"])
    if hi == lo:
        if "vmax" not in spec:
            hi = lo + max(abs(lo) * 1e-6, 1e-30)
        elif "vmin" not in spec:
            lo = hi - max(abs(hi) * 1e-6, 1e-30)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"Invalid common color limits for {spec['name']}")
    return {"norm": kind, "vmin": float(lo), "vmax": float(hi),
            "data_min": stats["min"], "data_max": stats["max"],
            "nonpositive_display_samples": stats["nonpositive_display_samples"]}


def _norm(limits):
    cls = LogNorm if limits["norm"] == "log" else Normalize
    return cls(vmin=limits["vmin"], vmax=limits["vmax"])


def _xlabel(config):
    kind = config["reference"]["kind"]
    return {"absolute": "z [um]", "linear": "z - linear reference [um]",
            "window_center": "z - window center [um]"}[kind]


def _title(iteration, time_s, zref):
    return f"iteration={iteration} | t={time_s * 1e12:.6g} ps | reference z={zref * 1e3:.6g} mm"


def _movies(paths_by_name, outdir, fps):
    import imageio.v2 as imageio
    for name, paths in paths_by_name.items():
        with imageio.get_writer(str(outdir / f"{name}.mp4"), fps=fps, macro_block_size=1) as writer:
            for path in paths:
                writer.append_data(imageio.imread(path))


def _versions():
    names = ("numpy", "h5py", "matplotlib", "openpmd-viewer", "imageio", "imageio-ffmpeg")
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def visualize(case_dir: str | Path, config: dict, outdir: str | Path, *,
              requested: list[int] | None, with_particles: bool = False,
              completed_log: str | Path | None = None, static_diagnostics: bool = False,
              min_age_seconds: float = 600.0, movie: bool = False, fps: float = 6.0) -> dict:
    validate_config(config)
    if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("min_age_seconds must be finite and nonnegative")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    case = Path(case_dir).resolve()
    output = Path(outdir).resolve()
    field_dir = (case / config["field_diag"]).resolve()
    particle_spec = config.get("particles") if with_particles else None
    if with_particles and particle_spec is None:
        raise ValueError("--with-particles requires an explicit particles configuration")
    particle_dir = (case / particle_spec["diag"]).resolve() if particle_spec else None
    for source_dir in (field_dir, particle_dir):
        if source_dir is not None and (output == source_dir or source_dir in output.parents):
            raise ValueError("Output must be outside raw diagnostic directories")
    log = completed_log or config.get("completed_log")
    if log is None and not static_diagnostics:
        raise ValueError("Supply a completed-step log, or --static-diagnostics for closed diagnostics")
    maximum = completed_step(case / log) if log is not None else None
    fields = inventory(field_dir, min_age_seconds=min_age_seconds, max_iteration=maximum)
    particles = inventory(particle_dir, min_age_seconds=min_age_seconds,
                          max_iteration=maximum) if particle_dir else None
    iterations = select_iterations(fields, particles, requested)
    # Exact file selection happens before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    (output / "field_frames").mkdir()
    if particle_spec:
        (output / "particle_histograms").mkdir()
        (output / "particle_frames").mkdir()
        energy_edges = bin_edges(particle_spec["energy_bins"])
        longitudinal_edges = bin_edges(particle_spec["longitudinal_bins"])
    scales = {spec["name"]: {"min": math.inf, "max": -math.inf,
                            "min_positive": math.inf, "nonpositive_display_samples": 0}
              for spec in config["fields"]}
    frames = []
    view_limits = {v["name"]: {"transverse_um": [math.inf, -math.inf],
                              "longitudinal_um": [math.inf, -math.inf]}
                   for v in config["views"]}
    particle_max = 0.0
    spectrum_min, spectrum_max = math.inf, 0.0
    for iteration in iterations:
        with snapshot_series(fields[iteration], iteration) as ts:
            index = list(map(int, ts.iterations)).index(iteration)
            time_s = float(ts.t[index])
            if not math.isfinite(time_s):
                raise ValueError("Non-finite snapshot time")
            zref = None
            for spec, view, data, r, z, reference in _quantities(ts, iteration, config, time_s):
                if zref is not None and not math.isclose(zref, reference, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("Requested fields have different reference coordinates")
                zref = reference
                _update_scale(scales[spec["name"]], data)
                for key, axis in (("transverse_um", r), ("longitudinal_um", z)):
                    lo, hi = view_limits[view["name"]][key]
                    view_limits[view["name"]][key] = [min(lo, float(axis[0] - (axis[1] - axis[0]) / 2)),
                                                     max(hi, float(axis[-1] + (axis[-1] - axis[-2]) / 2))]
        row = {"iteration": iteration, "time_s": time_s, "reference_z_m": zref}
        if particle_spec:
            # Each species is read once per frame. Only small charge histograms persist.
            from .particle_histograms import histograms
            with snapshot_series(particles[iteration], iteration) as pts:
                index = list(map(int, pts.iterations)).index(iteration)
                if not math.isclose(float(pts.t[index]), time_s, rel_tol=1e-10, abs_tol=1e-21):
                    raise ValueError(f"Field and particle timestamps differ at {iteration}")
                spectrum, phase, rows = histograms(pts, iteration, particle_spec, zref,
                                                   energy_edges, longitudinal_edges)
            np.savez_compressed(output / "particle_histograms" / f"{iteration:08d}.npz",
                                energy_edges_mev=energy_edges, longitudinal_edges_um=longitudinal_edges,
                                charge_spectrum_pc=spectrum, charge_phase_pc=phase)
            row["electron_species"] = rows
            particle_max = max(particle_max, float(phase.max()))
            density = spectrum / np.diff(energy_edges)
            spectrum_max = max(spectrum_max, float(density.max()))
            if np.any(density > 0):
                spectrum_min = min(spectrum_min, float(density[density > 0].min()))
        frames.append(row)
        print(f"[SCALE] iteration={iteration}", flush=True)
    limits = {f["name"]: _limits(f, scales[f["name"]]) for f in config["fields"]}
    paths = {f"fields_{v['name']}": [] for v in config["views"]}
    if particle_spec:
        paths["electrons"] = []
    for row in frames:
        iteration, time_s, zref = row["iteration"], row["time_s"], row["reference_z_m"]
        figures = {}
        for view in config["views"]:
            fig, axes = plt.subplots(len(config["fields"]), 1, squeeze=False,
                                      figsize=(11, 3.4 * len(config["fields"])),
                                      constrained_layout=True)
            figures[view["name"]] = (fig, axes[:, 0])
        try:
            with snapshot_series(fields[iteration], iteration) as ts:
                for spec, view, data, r, z, current_reference in _quantities(ts, iteration, config, time_s):
                    if not math.isclose(zref, current_reference, rel_tol=1e-12, abs_tol=1e-15):
                        raise ValueError("Reference changed between visualization passes")
                    fig, axes = figures[view["name"]]
                    ax = axes[config["fields"].index(spec)]
                    scale = limits[spec["name"]]
                    shown = np.ma.masked_less_equal(data, 0) if scale["norm"] == "log" else data
                    artist = ax.pcolormesh(z, r, shown, norm=_norm(scale),
                                           cmap="RdBu_r" if scale["norm"] == "symmetric" else "viridis",
                                           shading="nearest", rasterized=True)
                    ax.set_aspect("equal")
                    ax.set(xlabel=_xlabel(config), ylabel=f"signed transverse coordinate [um], theta={view['theta_rad']:g}",
                           xlim=view["longitudinal_um"] or view_limits[view["name"]]["longitudinal_um"],
                           ylim=view["transverse_um"] or view_limits[view["name"]]["transverse_um"],
                           title=f"{spec.get('label', spec['name'])} | m={spec['m']}")
                    fig.colorbar(artist, ax=ax, label=spec.get("unit", "SI"))
            for name, (fig, axes) in figures.items():
                fig.suptitle(_title(iteration, time_s, zref) + f" | {name}")
                path = output / "field_frames" / f"{name}_{iteration:08d}.png"
                fig.savefig(path, dpi=120)
                paths[f"fields_{name}"].append(path)
        finally:
            for fig, _ in figures.values():
                plt.close(fig)
        if particle_spec:
            path = output / "particle_histograms" / f"{iteration:08d}.npz"
            with np.load(path) as arrays:
                spectrum, phase = arrays["charge_spectrum_pc"], arrays["charge_phase_pc"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            try:
                axes[0].stairs(spectrum / np.diff(energy_edges), energy_edges)
                if spectrum_max > 0:
                    axes[0].set_yscale("log")
                    axes[0].set_ylim(spectrum_min * 0.8, spectrum_max * 1.2)
                else:
                    axes[0].set_ylim(0, 1)
                if particle_spec["energy_bins"].get("spacing") == "log":
                    axes[0].set_xscale("log")
                    axes[1].set_yscale("log")
                axes[0].set(xlabel="Electron kinetic energy [MeV]", ylabel="dQ/dK [pC/MeV]",
                            xlim=(energy_edges[0], energy_edges[-1]), title="Window electron spectrum")
                pnorm = LogNorm(vmin=max(particle_max * 1e-6, 1e-30),
                                vmax=max(particle_max, 1e-29))
                artist = axes[1].pcolormesh(longitudinal_edges, energy_edges,
                                            np.ma.masked_less_equal(phase.T, 0),
                                            norm=pnorm, cmap="magma", shading="flat")
                axes[1].set(xlabel=_xlabel(config), ylabel="Electron kinetic energy [MeV]",
                            title="Window population, integrated over transverse coordinates")
                fig.colorbar(artist, ax=axes[1], label="Electron charge magnitude [pC/bin]")
                selection = "uz > 0" if particle_spec.get("forward_only", False) else "all directions"
                fig.suptitle(_title(iteration, time_s, zref) + "\n" +
                             ", ".join(particle_spec["species"]) +
                             f" | {selection}, K >= {particle_spec.get('energy_min_mev', 0):g} MeV")
                path = output / "particle_frames" / f"electrons_{iteration:08d}.png"
                fig.savefig(path, dpi=120)
                paths["electrons"].append(path)
            finally:
                plt.close(fig)
        print(f"[FRAME] iteration={iteration}", flush=True)
    if movie:
        _movies(paths, output, fps)
    for mapping in (fields, particles):
        if mapping is not None:
            for iteration in iterations:
                mapping[iteration].check()
    metadata = {
        "schema_version": 1, "status": "complete", "case_dir": str(case),
        "config": config, "particles_enabled": bool(particle_spec),
        "completed_step_at_selection": maximum, "static_diagnostics": static_diagnostics,
        "min_age_seconds": min_age_seconds, "frames": frames, "field_scales": limits,
        "view_limits_um": view_limits,
        "particle_scales": ({"phase_charge_max_pc_per_bin": particle_max,
                             "spectrum_min_positive_pc_per_mev": spectrum_min if spectrum_max > 0 else None,
                             "spectrum_max_pc_per_mev": spectrum_max} if particle_spec else None),
        "sources": {str(it): {"fields": fields[it].metadata(),
                              **({"particles": particles[it].metadata()} if particles else {})}
                    for it in iterations},
        "versions": _versions(), "fps": fps if movie else None,
        "movie_time": "uniform display cadence; consult each frame's physical timestamp",
        "scope": "saved moving-window populations; no trapping, guiding, exit or convergence assertion",
        "transverse_e2": "smoothed total transverse electric field squared; not isolated laser intensity or a0",
        "density": "electron_density converts the configured species rho by -rho/e, without ion charge",
        "log_masking": "nonpositive values are masked; counts cover requested views including overlaps",
        "particle_weights": "w counts physical electrons; Q=e*sum(w); no extra cylindrical factor",
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")},
    }
    (output / "visualization.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
