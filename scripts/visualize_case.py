#!/usr/bin/env python3
"""Visualize existing RZ fields and optional window electrons, without campaign scores."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laser_guiding.config import load_config
from laser_guiding.visualization import visualize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path,
                        help="A new directory outside raw diagnostics")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--iterations", help="Comma-separated exact iterations")
    choice.add_argument("--all-frames", action="store_true",
                        help="Use all eligible frames; fields/particles must agree when both are requested")
    parser.add_argument("--with-particles", action="store_true",
                        help="Enable configured electron histograms; default is fields only")
    parser.add_argument("--completed-log", type=Path,
                        help="Override completed-step log; relative to case directory")
    parser.add_argument("--static-diagnostics", action="store_true",
                        help="Declare diagnostics closed when no completed-step log is supplied")
    parser.add_argument("--min-age-seconds", type=float, default=600)
    parser.add_argument("--movie", action="store_true", help="Also encode MP4 files")
    parser.add_argument("--fps", type=float, default=6)
    args = parser.parse_args()
    requested = None
    if args.iterations is not None:
        try:
            requested = [int(value.strip()) for value in args.iterations.split(",")]
        except ValueError:
            parser.error("--iterations must contain comma-separated integers")
    config = load_config(args.config, args.case_dir)
    visualize(args.case_dir, config, args.outdir, requested=requested,
              with_particles=args.with_particles, completed_log=args.completed_log,
              static_diagnostics=args.static_diagnostics,
              min_age_seconds=args.min_age_seconds, movie=args.movie, fps=args.fps)
    print(f"[OK] {args.outdir / 'visualization.json'}")


if __name__ == "__main__":
    main()
