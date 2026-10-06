#!/usr/bin/env python3
"""Reduce a case or campaign to physical beam evolution products (no BO score)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.beam_evolution import (
    BeamEvolutionConfig, resolve_frame_coordinate, summarize_beam_evolution,
    summarize_beam_frame, unavailable_beam_frame, write_beam_csv,
)
from cap_guiding.diagnostics import resolve_particle_diag_dir
from cap_guiding.openpmd_io import open_series
from cap_guiding.particle_exit import read_resolved_particle_exit_target
from cap_guiding.particles import read_particle_dump


def read_plasma_context(
    path: Path, *, start_um: float | None, end_um: float | None,
    exit_iteration: int | None,
) -> tuple[float, float, int | None, str]:
    """Use proven resolved_parameters keys in metres; never infer a ramp or origin."""
    if (start_um is None) != (end_um is None):
        raise ValueError("Specify both --plasma-start-um and --plasma-end-um")
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("resolved_parameters must be a JSON object")
    has_bounds = "plasma_start_z" in payload and "plasma_end_z" in payload
    if start_um is None:
        if not has_bounds:
            raise ValueError("Need resolved plasma_start_z/plasma_end_z or explicit absolute bounds in um")
        start_um = float(payload["plasma_start_z"]) * 1e6
        end_um = float(payload["plasma_end_z"]) * 1e6
        source = str(path.resolve())
    else:
        source = "explicit_um"
        if has_bounds and not all(math.isclose(a, float(payload[k]) * 1e6, rel_tol=1e-10, abs_tol=1e-6)
                                  for a, k in ((start_um, "plasma_start_z"), (end_um, "plasma_end_z"))):
            raise ValueError("Explicit bounds disagree with resolved plasma bounds")
    if not (math.isfinite(start_um) and math.isfinite(end_um) and end_um > start_um):
        raise ValueError("Plasma bounds must be finite with end > start")
    targets = payload.get("particle_diagnostic_targets", {})
    if isinstance(targets, dict) and "capillary_exit" in targets:
        target = read_resolved_particle_exit_target(path, exit_kind="capillary")
        authoritative = target["target_particle_iteration"]
        if exit_iteration is not None and exit_iteration != authoritative:
            raise ValueError("Explicit exit iteration disagrees with resolved exact target")
        exit_iteration = authoritative
    if exit_iteration is not None and exit_iteration < 0:
        raise ValueError("Exit iteration must be non-negative")
    return start_um, end_um, exit_iteration, source


def read_guiding_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"iteration", "z_max_um"}.issubset(reader.fieldnames or ()):
            raise ValueError("Guiding CSV must contain iteration and absolute z_max_um")
        rows = list(reader)
    if not rows:
        raise ValueError("Guiding CSV is empty")
    iterations = [int(r["iteration"]) for r in rows]
    if len(set(iterations)) != len(iterations):
        raise ValueError("Duplicate guiding iterations")
    return rows


def analyze_case(case_dir: Path, args: argparse.Namespace) -> bool:
    outdir = Path(args.outdir) if args.outdir else case_dir / args.outdir_name
    products = [outdir / "beam_evolution.csv", outdir / "beam_evolution_summary.csv"]
    if not args.overwrite and any(p.exists() for p in products):
        raise FileExistsError("Beam products exist; use --overwrite (legacy outputs are never touched)")
    resolved = Path(args.resolved_parameters) if args.resolved_parameters else case_dir / "resolved_parameters.json"
    start, end, exit_iteration, bounds_source = read_plasma_context(
        resolved, start_um=args.plasma_start_um, end_um=args.plasma_end_um,
        exit_iteration=args.exit_iteration,
    )
    guiding_path = Path(args.guiding_metrics) if args.guiding_metrics else case_dir / "guiding_metrics.csv"
    guiding = read_guiding_rows(guiding_path)
    species = args.species.replace(",", " ").split()
    if not species or len(set(species)) != len(species):
        raise ValueError("Species list must be non-empty without duplicates")
    overrides: dict[str, Path] = {}
    for item in args.species_diag:
        name, separator, directory = item.partition("=")
        if not separator or name not in species or not directory or name in overrides:
            raise ValueError("--species-diag must be a unique requested SPECIES=PATH")
        overrides[name] = Path(directory)
    cfg = BeamEvolutionConfig(args.energy_threshold_mev, args.min_effective_particles)
    series: dict[str, Any] = {}
    diagnostics: dict[str, Path] = {}
    setup_errors: dict[str, str] = {}
    available: dict[str, set[int]] = {}
    for name in species:
        try:
            diag = overrides[name] if name in overrides else resolve_particle_diag_dir(case_dir, name)
            diagnostics[name] = diag
            series[name] = open_series(diag)
            available[name] = {int(it) for it in series[name].iterations}
        except Exception as exc:
            setup_errors[name] = f"{type(exc).__name__}: {exc}"
            available[name] = set()
    iterations = sorted(set().union(*available.values()))
    if not iterations:
        raise RuntimeError(f"No readable particle iterations: {setup_errors}")
    if exit_iteration is not None and exit_iteration not in iterations:
        iterations.append(exit_iteration)
        iterations.sort()
    # Preserve union-of-species iterations: a missing dump in one species is an
    # unavailable row, never an empty registered species.
    by_species: dict[str, list[dict[str, Any]]] = {name: [] for name in species}
    errors = bool(setup_errors)
    for iteration in iterations:
        for name in species:
            ts = series.get(name)
            time_fs = float("nan")
            if ts is not None and iteration in available[name]:
                index = list(map(int, ts.iterations)).index(iteration)
                time_fs = float(np.asarray(ts.t)[index]) * 1e15
            else:
                # Timestamp is metadata only. No particle data are fabricated.
                match = next((r for r in guiding if int(r["iteration"]) == iteration), None)
                if match is not None:
                    time_fs = float(match.get("time_fs", "nan"))
            try:
                z, source = resolve_frame_coordinate(
                    guiding, iteration=iteration, time_fs=time_fs,
                    max_coordinate_gap_um=args.max_coordinate_gap_um,
                )
            except ValueError as exc:
                z, source = float("nan"), "unavailable"
                coordinate_error = str(exc)
            else:
                coordinate_error = "" if math.isfinite(z) else "No bracketed guiding coordinate"
            try:
                if iteration not in available[name]:
                    raise RuntimeError(setup_errors.get(name, "Particle iteration is missing for this species"))
                dump = read_particle_dump(diagnostics[name], species=name, iteration=iteration, series=ts)
                row = summarize_beam_frame(dump, species_scope=name, z_frame_um=z,
                                           coordinate_source=source, config=cfg)
                if row["population_status"] == "invalid_particle_data":
                    errors = True
            except Exception as exc:
                row = unavailable_beam_frame(
                    iteration=iteration, time_fs=time_fs, species_scope=name,
                    z_frame_um=z, coordinate_source=source,
                    error=f"{type(exc).__name__}: {exc}", config=cfg,
                )
                errors = True
                print(f"[UNAVAILABLE] {case_dir.name} {name} {iteration}: {row['read_error']}")
            if coordinate_error:
                row["read_error"] = "; ".join(filter(None, (row["read_error"], coordinate_error)))
                errors = True
            row["case_name"] = case_dir.name
            by_species[name].append(row)
        print(f"[FRAME] {case_dir.name} iteration={iteration}")
    summaries = []
    for name in species:
        summary = summarize_beam_evolution(
            by_species[name], plasma_start_um=start, plasma_end_um=end,
            max_gap_um=args.max_gap_um, exit_iteration=exit_iteration,
        )
        summary["plasma_bounds_source"] = bounds_source
        summary["guiding_metrics_source"] = str(guiding_path.resolve())
        summary["particle_diagnostic_source"] = str(diagnostics[name].resolve()) if name in diagnostics else "unavailable"
        summaries.append(summary)
    frames = sorted((r for rows in by_species.values() for r in rows),
                    key=lambda r: (int(r["iteration"]), str(r["species_scope"])))
    write_beam_csv(frames, products[0], frames=True)
    write_beam_csv(summaries, products[1])
    partial = any(s["integration_status"] != "complete" or s["exit_snapshot_status"] in
                  ("missing", "unavailable", "invalid_particle_data") for s in summaries)
    print(f"[{'PARTIAL' if errors or partial else 'OK'}] {case_dir.name}: {products}")
    return not (errors or partial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case-dir")
    mode.add_argument("--campaign-root")
    parser.add_argument("--case-glob", default="[0-9][0-9][0-9]_*")
    parser.add_argument("--species", default="electrons")
    parser.add_argument("--species-diag", action="append", default=[], metavar="SPECIES=PATH")
    parser.add_argument("--guiding-metrics")
    parser.add_argument("--resolved-parameters")
    parser.add_argument("--plasma-start-um", type=float)
    parser.add_argument("--plasma-end-um", type=float)
    parser.add_argument("--exit-iteration", type=int)
    parser.add_argument("--energy-threshold-mev", type=float, default=50.0)
    parser.add_argument("--min-effective-particles", type=float, default=3.0)
    parser.add_argument("--max-gap-um", type=float,
                        help="Maximum distance between particle frames to integrate; default 1.5 times median spacing")
    parser.add_argument("--max-coordinate-gap-um", type=float,
                        help="Maximum guiding bracket distance for time interpolation")
    parser.add_argument("--outdir")
    parser.add_argument("--outdir-name", default="beam_analysis")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.campaign_root and any((args.species_diag, args.guiding_metrics, args.resolved_parameters, args.outdir)):
        parser.error("Per-case diagnostic/metadata/output overrides are only valid with --case-dir")
    cases = [Path(args.case_dir)] if args.case_dir else sorted(
        p for p in Path(args.campaign_root).glob(args.case_glob) if p.is_dir()
    )
    if not cases:
        parser.error("No matching cases")
    ok = failed = 0
    for case in cases:
        try:
            success = analyze_case(case, args)
        except Exception as exc:
            print(f"[FAIL] {case}: {type(exc).__name__}: {exc}")
            success = False
        ok += int(success)
        failed += int(not success)
    print(f"[SUMMARY] complete={ok} partial_or_failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
