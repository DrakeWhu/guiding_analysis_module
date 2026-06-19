#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.openpmd_io import open_series, get_iterations, describe_series
from cap_guiding.particles import (
    last_iteration,
    read_particle_dump,
    save_energy_spectrum,
    save_longitudinal_phase_space,
    save_longitudinal_energy_space,
    summarize_dump,
    write_summary_csv,
    summarize_acceptance_curves,
    write_acceptance_curves_csv,
)


def parse_case_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}

    if not path.exists():
        return env

    for raw in path.read_text().splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            parts = shlex.split(value)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip("'\"")

        env[key] = value

    return env


def get_float_env(env: dict[str, str], names: list[str]) -> float | None:
    for name in names:
        if name in env and env[name] != "":
            try:
                return float(env[name])
            except ValueError:
                pass
    return None


def infer_plateau_length_from_case_name(case_dir: Path) -> float | None:
    match = re.search(r"_L([0-9]+(?:[p.][0-9]+)?)mm_", case_dir.name)
    if not match:
        return None

    return float(match.group(1).replace("p", "."))


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in str(text).replace(",", " ").split():
        values.append(float(raw))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def case_id_from_case_dir(case_dir: Path) -> str:
    return case_dir.name.split("_", 1)[0]


def target_propagation_from_case(
    *,
    case_dir: Path,
    exit_kind: str,
    target_propagation_mm: float | None,
    downramp_mm: float | None = None,
) -> float:
    if target_propagation_mm is not None:
        return float(target_propagation_mm)

    env = parse_case_env(case_dir / "case.env")

    plateau_mm = get_float_env(
        env,
        [
            "PLATEAU_LENGTH_MM",
            "plateau_length_mm",
            "L_PLATEAU_MM",
            "LENGTH_MM",
        ],
    )

    if plateau_mm is None:
        plateau_m = get_float_env(
            env,
            [
                "CAP_PLATEAU_LENGTH_M",
                "PLATEAU_LENGTH_M",
                "plateau_length_m",
            ],
        )
        if plateau_m is not None:
            plateau_mm = plateau_m * 1.0e3

    if plateau_mm is None:
        plateau_mm = infer_plateau_length_from_case_name(case_dir)

    if plateau_mm is None:
        raise ValueError(
            f"Could not infer plateau length from {case_dir / 'case.env'} "
            f"or case name {case_dir.name!r}. "
            "Use --target-propagation-mm explicitly."
        )

    if exit_kind == "plateau":
        return float(plateau_mm)

    if exit_kind == "capillary":
        # Conservative default: capillary exit = plateau exit unless an
        # explicit downramp/capillary length is present in case.env.

        if downramp_mm is not None:
            return float(plateau_mm + float(downramp_mm))

        capillary_mm = get_float_env(
            env,
            [
                "CAPILLARY_LENGTH_MM",
                "capillary_length_mm",
                "TOTAL_CAPILLARY_LENGTH_MM",
            ],
        )
        if capillary_mm is not None:
            return float(capillary_mm)

        downramp_mm = get_float_env(
            env,
            [
                "DOWNRAMP_LENGTH_MM",
                "RAMP_DOWN_LENGTH_MM",
                "downramp_length_mm",
            ],
        )
        if downramp_mm is not None:
            return float(plateau_mm + downramp_mm)

        print(
            "[WARN] --exit-kind capillary requested, but no explicit capillary "
            "or downramp length was found in case.env. Falling back to plateau exit."
        )
        return float(plateau_mm)

    raise ValueError(f"Unknown exit_kind: {exit_kind}")


def read_guiding_iteration_at_propagation(
    metrics_csv: Path,
    *,
    target_propagation_mm: float,
) -> dict[str, Any]:
    df = pd.read_csv(metrics_csv)

    required = ["iteration", "propagation_mm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {missing}")

    df = df.copy()
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["propagation_mm"] = pd.to_numeric(df["propagation_mm"], errors="coerce")
    df = df.dropna(subset=["iteration", "propagation_mm"])

    if df.empty:
        raise ValueError(f"No valid iteration/propagation rows in {metrics_csv}")

    idx = (df["propagation_mm"] - float(target_propagation_mm)).abs().idxmin()
    row = df.loc[idx]

    return {
        "target_propagation_mm": float(target_propagation_mm),
        "target_guiding_iteration": int(row["iteration"]),
        "target_guiding_propagation_mm": float(row["propagation_mm"]),
    }


def nearest_particle_iteration(
    *,
    diag: Path,
    target_iteration: int,
) -> dict[str, Any]:
    ts = open_series(diag)
    particle_iterations = list(map(int, get_iterations(ts, stride=1)))

    if not particle_iterations:
        raise RuntimeError(f"No particle iterations found in {diag}")

    arr = np.asarray(particle_iterations, dtype=int)
    selected = int(arr[np.argmin(np.abs(arr - int(target_iteration)))])

    return {
        "selected_particle_iteration": selected,
        "target_iteration_delta": int(selected - int(target_iteration)),
        "available_particle_iterations_min": int(arr.min()),
        "available_particle_iterations_max": int(arr.max()),
        "n_available_particle_iterations": int(arr.size),
    }


def resolve_iterations(
    *,
    diag: Path,
    which: str,
    stride: int,
    case_dir: Path,
    guiding_metrics: Path | None,
    exit_kind: str,
    target_propagation_mm: float | None,
    downramp_mm: float | None,
) -> tuple[list[int], dict[str, Any]]:
    if which == "last":
        iteration = last_iteration(diag)
        return [iteration], {
            "selection_mode": "last",
            "selected_particle_iteration": int(iteration),
        }

    if which == "all":
        ts = open_series(diag)
        iterations = get_iterations(ts, stride=stride)
        return list(map(int, iterations)), {
            "selection_mode": "all",
            "analysis_stride": int(stride),
        }

    if which == "exit":
        metrics_csv = (
            guiding_metrics
            if guiding_metrics is not None
            else case_dir / "guiding_metrics.csv"
        )

        if not metrics_csv.exists():
            raise FileNotFoundError(
                f"Missing guiding metrics for --which exit: {metrics_csv}"
            )

        target_mm = target_propagation_from_case(
            case_dir=case_dir,
            exit_kind=exit_kind,
            target_propagation_mm=target_propagation_mm,
            downramp_mm=downramp_mm,
        )

        guiding_info = read_guiding_iteration_at_propagation(
            metrics_csv,
            target_propagation_mm=target_mm,
        )

        particle_info = nearest_particle_iteration(
            diag=diag,
            target_iteration=guiding_info["target_guiding_iteration"],
        )

        selection_info = {
            "selection_mode": "exit",
            "exit_kind": exit_kind,
            "guiding_metrics_csv": str(metrics_csv),
            **guiding_info,
            **particle_info,
        }

        return [int(particle_info["selected_particle_iteration"])], selection_info

    raise ValueError(f"Unknown --which mode: {which}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze WarpX electron particle openPMD diagnostics for LWFA signatures."
    )
    parser.add_argument(
        "--diag",
        required=True,
        help="Path to particle openPMD diagnostic directory, e.g. CASE/diags/electron_particles",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory, usually CASE/particle_analysis",
    )
    parser.add_argument("--species", default="electrons")
    parser.add_argument("--which", choices=["last", "all", "exit"], default="last")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hot-energy-mev", type=float, default=10.0)
    parser.add_argument(
        "--longitudinal",
        choices=["x", "y", "z"],
        default="z",
        help="Longitudinal coordinate/momentum component for phase space and forward cut.",
    )
    parser.add_argument(
        "--no-forward-cut",
        action="store_true",
        help="Do not require positive longitudinal momentum for hot-electron selection.",
    )
    parser.add_argument(
        "--exit-window-mm",
        type=float,
        default=None,
        help="Optional particle-space window behind max longitudinal coordinate in selected dump.",
    )
    parser.add_argument(
        "--exit-kind",
        choices=["plateau", "capillary"],
        default="plateau",
        help="Physical target used by --which exit.",
    )
    parser.add_argument(
        "--target-propagation-mm",
        type=float,
        default=None,
        help="Override physical target propagation in mm for --which exit.",
    )
    parser.add_argument(
        "--guiding-metrics",
        default=None,
        help="Optional guiding_metrics.csv. Defaults to CASE_DIR/guiding_metrics.csv.",
    )
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--emax-mev", type=float, default=None)
    parser.add_argument(
        "--spectrum-emin-mev",
        type=float,
        default=0.0,
        help="Minimum kinetic energy shown in the energy spectrum plot.",
    )
    parser.add_argument(
        "--spectrum-log-y",
        action="store_true",
        help="Use logarithmic y axis for energy spectrum plots.",
    )
    parser.add_argument("--max-phase-points", type=int, default=200_000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--downramp-mm",
        type=float,
        default=None,
        help=(
            "Downramp length in mm used when --which exit --exit-kind capillary "
            "and no explicit total capillary length is available."
        ),
    )
    parser.add_argument(
        "--acceptance-theta-cuts-mrad",
        type=parse_float_list,
        default=parse_float_list("2,5,10,20,50"),
        help=(
            "Comma- or whitespace-separated theta_r cuts in mrad for "
            "particle_acceptance_curves.csv."
        ),
    )
    parser.add_argument(
        "--acceptance-energy-cuts-mev",
        type=parse_float_list,
        default=parse_float_list("10,25,50,100,150,200,250,300"),
        help=(
            "Comma- or whitespace-separated minimum kinetic energies in MeV "
            "for particle_acceptance_curves.csv."
        ),
    )
    args = parser.parse_args()

    diag = Path(args.diag)
    outdir = Path(args.outdir)
    case_dir = diag.parents[1] if diag.parent.name == "diags" else outdir.parent
    guiding_metrics = Path(args.guiding_metrics) if args.guiding_metrics else None

    summary_csv = outdir / "particle_summary.csv"
    acceptance_csv = outdir / "particle_acceptance_curves.csv"

    write_summary = args.overwrite or not summary_csv.exists()
    write_acceptance = args.overwrite or not acceptance_csv.exists()

    if (
        args.skip_existing
        and not args.overwrite
        and not write_summary
        and not write_acceptance
    ):
        print(f"[SKIP] existing {summary_csv} and {acceptance_csv}")
        return

    if not args.skip_existing and not args.overwrite:
        existing = [str(p) for p in (summary_csv, acceptance_csv) if p.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(existing)
                + ". Use --overwrite or --skip-existing."
            )

    print("=== Particle case analysis ===")
    print(f"case_dir          = {case_dir}")
    print(f"diag              = {diag}")
    print(f"outdir            = {outdir}")
    print(f"species           = {args.species}")
    print(f"which             = {args.which}")
    print(f"exit_kind         = {args.exit_kind}")
    print(f"target_prop_mm    = {args.target_propagation_mm}")
    print(f"guiding_metrics   = {guiding_metrics or case_dir / 'guiding_metrics.csv'}")
    print(f"stride            = {args.stride}")
    print(f"hot_energy_mev    = {args.hot_energy_mev}")
    print(f"longitudinal      = {args.longitudinal}")
    print(f"forward_cut       = {not args.no_forward_cut}")
    print(f"exit_window_mm    = {args.exit_window_mm}")
    print(f"downramp_mm      = {args.downramp_mm}")
    print(f"spectrum_emin    = {args.spectrum_emin_mev}")
    print(f"spectrum_log_y   = {args.spectrum_log_y}")
    print(f"accept_theta_mrad= {args.acceptance_theta_cuts_mrad}")
    print(f"accept_Emin_MeV  = {args.acceptance_energy_cuts_mev}")
    print("==============================")

    ts = open_series(diag)
    print("[SERIES]", describe_series(ts))

    iterations, selection_info = resolve_iterations(
        diag=diag,
        which=args.which,
        stride=args.stride,
        case_dir=case_dir,
        guiding_metrics=guiding_metrics,
        exit_kind=args.exit_kind,
        target_propagation_mm=args.target_propagation_mm,
        downramp_mm=args.downramp_mm,
    )

    if not iterations:
        raise RuntimeError(f"No particle iterations selected in {diag}")

    print("[SELECTION]")
    for key, value in selection_info.items():
        print(f"  {key} = {value}")

    rows = []
    acceptance_rows = []
    plots_dir = outdir / "plots"
    case_id = case_id_from_case_dir(case_dir)
    case_name = case_dir.name

    for iteration in iterations:
        print(f"[READ] iteration {iteration}")

        dump = read_particle_dump(
            diag,
            species=args.species,
            iteration=iteration,
        )

        row = summarize_dump(
            dump,
            hot_energy_mev=args.hot_energy_mev,
            longitudinal=args.longitudinal,
            exit_window_mm=args.exit_window_mm,
            forward_only=not args.no_forward_cut,
        )

        row = {
            **selection_info,
            **row,
        }
        rows.append(row)

        acceptance_rows.extend(
            summarize_acceptance_curves(
                dump,
                case_id=case_id,
                case_name=case_name,
                selection_mode=str(selection_info.get("selection_mode", "")),
                selected_particle_iteration=int(
                    selection_info.get("selected_particle_iteration", iteration)
                ),
                theta_cuts_mrad=args.acceptance_theta_cuts_mrad,
                e_min_mev=args.acceptance_energy_cuts_mev,
                longitudinal=args.longitudinal,
                forward_only=not args.no_forward_cut,
            )
        )

        suffix = f"it{int(iteration):08d}"

        save_energy_spectrum(
            dump,
            path=plots_dir / f"energy_spectrum_{suffix}.png",
            hot_energy_mev=args.hot_energy_mev,
            bins=args.bins,
            emax_mev=args.emax_mev,
            spectrum_min_energy_mev=args.spectrum_emin_mev,
            log_y=args.spectrum_log_y,
        )

        try:
            save_longitudinal_phase_space(
                dump,
                path=plots_dir / f"longitudinal_phase_space_hot_{suffix}.png",
                hot_energy_mev=args.hot_energy_mev,
                longitudinal=args.longitudinal,
                exit_window_mm=args.exit_window_mm,
                forward_only=not args.no_forward_cut,
                max_points=args.max_phase_points,
            )
            save_longitudinal_energy_space(
                dump,
                path=plots_dir / f"longitudinal_energy_space_hot_{suffix}.png",
                hot_energy_mev=args.hot_energy_mev,
                longitudinal=args.longitudinal,
                exit_window_mm=args.exit_window_mm,
                forward_only=not args.no_forward_cut,
                max_points=args.max_phase_points,
            )
        except ValueError as exc:
            print(f"[WARN] skipping phase-space plot for iteration {iteration}: {exc}")

    if write_summary:
        write_summary_csv(rows, summary_csv)
        print(f"[OK] wrote {summary_csv}")
    else:
        print(f"[USE] existing {summary_csv}")

    if write_acceptance:
        write_acceptance_curves_csv(acceptance_rows, acceptance_csv)
        print(f"[OK] wrote {acceptance_csv}")
    else:
        print(f"[USE] existing {acceptance_csv}")


if __name__ == "__main__":
    main()
