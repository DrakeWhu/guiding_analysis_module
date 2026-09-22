#!/usr/bin/env python3
"""Run the Python reference pipeline on the synthetic fixtures.

Outputs land in cpp/tests/data/golden/:
- cases/<case>/: scripts/analyze_case.py (guiding_metrics.csv + single-case score)
- campaign/: scripts/analyze_campaign.py --run-cases --run-triplets (reports,
  case metrics and triplet tables) plus its stdout
- particles/<run>/: scripts/analyze_particle_case.py for every entry of
  particles/runs.json (CSV products and stdout; plots are deleted)
- particles/campaign/: scripts/analyze_particle_campaign.py on a copy of the
  synthetic campaign (per-case CSV products and stdout)
- scoring/<run>/: score_campaign, score_triplets, score_beamlike_pairs and
  compare_guiding_beamlike_scores on synthetic/scoring (make_scoring_fixtures.py),
  run from cpp/tests/data so recorded paths are relative to it
metadata.json records the reference commit and library versions.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "cpp" / "tests" / "data"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def run_reference(*args: str, check: bool = True, cwd: Path = REPO) -> str:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        # Unbuffered so parent and child-process lines keep their real order.
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO),
             "PYTHONWARNINGS": "ignore"},
    )
    return result.stdout


CHANNEL = "synthetic/campaign/000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz"
UNIFORM = "synthetic/campaign/001_f20_uni_n4e18cm3_L2mm_refd150um_foc0um_rz"
CHANNEL_METRICS = "golden/cases/000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz/guiding_metrics.csv"
UNIFORM_METRICS = "golden/cases/001_f20_uni_n4e18cm3_L2mm_refd150um_foc0um_rz/guiding_metrics.csv"
# Option values that are paths relative to cpp/tests/data.
PATH_OPTIONS = {"guiding-metrics", "resolved-parameters"}

# analyze_particle_case.py runs; options map flag names to values (True for a
# bare flag). The C++ parity test reads the same list from runs.json.
PARTICLE_RUNS = [
    {"name": "chan_last", "diag": f"{CHANNEL}/diags/plasma_electrons", "options": {}},
    {
        "name": "chan_multispecies_all_window",
        "diag": f"{CHANNEL}/diags/plasma_electrons",
        "options": {"species": "electrons,ionized_electrons", "which": "all", "exit-window-mm": 0.03},
    },
    {
        "name": "chan_exit_legacy_resolved_target",
        "diag": f"{CHANNEL}/diags/plasma_electrons",
        "options": {"which": "exit", "guiding-metrics": CHANNEL_METRICS, "maximum-target-iteration-delta": 100},
    },
    {
        "name": "chan_exit_exact_capillary",
        "diag": f"{CHANNEL}/diags/plasma_electrons",
        "options": {
            "species": "ionized_electrons electrons",
            "which": "exit",
            "exit-kind": "capillary",
            "resolved-parameters": f"{CHANNEL}/resolved_parameters.json",
            "guiding-metrics": CHANNEL_METRICS,
        },
    },
    {
        "name": "chan_exit_exact_without_guiding",
        "diag": f"{CHANNEL}/diags/plasma_electrons",
        "options": {"which": "exit", "resolved-parameters": f"{CHANNEL}/resolved_parameters.json"},
    },
    {
        "name": "uni_last_custom_cuts",
        "diag": f"{UNIFORM}/diags/electron_particles",
        "options": {
            "hot-energy-mev": 20.0,
            "no-forward-cut": True,
            "acceptance-theta-cuts-mrad": "3, 1",
            "acceptance-energy-cuts-mev": "40 5",
            "soft50-energy-low-mev": 15.0,
            "soft50-energy-target-mev": 60.0,
            "soft50-curve-energy-low-mev": "20,5,15",
        },
    },
    {
        "name": "uni_all_stride2_longitudinal_x",
        "diag": f"{UNIFORM}/diags/electron_particles",
        "options": {"which": "all", "stride": 2, "longitudinal": "x", "exit-window-mm": 0.05},
    },
    {
        "name": "uni_exit_case_env_capillary",
        "diag": f"{UNIFORM}/diags/electron_particles",
        "options": {"which": "exit", "exit-kind": "capillary", "guiding-metrics": UNIFORM_METRICS},
    },
    {
        "name": "uni_exit_explicit_target_tie",
        "diag": f"{UNIFORM}/diags/electron_particles",
        "options": {"which": "exit", "target-propagation-mm": 5.95, "guiding-metrics": UNIFORM_METRICS},
    },
]


def particle_run_arguments(run: dict) -> list[str]:
    arguments = ["--diag", rel(DATA / run["diag"])]
    for flag, value in run["options"].items():
        if value is True:
            arguments.append(f"--{flag}")
        elif flag in PATH_OPTIONS:
            arguments += [f"--{flag}", rel(DATA / value)]
        else:
            arguments += [f"--{flag}", str(value)]
    return arguments


SCORING_CAMPAIGN = ["--campaign-root", "synthetic/scoring/campaign", "--case-metrics-root", "synthetic/scoring/case_metrics"]

# Script runs of the scoring layers; options are passed verbatim (paths relative
# to cpp/tests/data) and read back by the C++ parity test from runs.json.
SCORING_RUNS = [
    {"name": "campaign_all", "script": "score_campaign", "options": {"case-type": "all", "top": 10}},
    {
        "name": "campaign_channel_custom",
        "script": "score_campaign",
        "options": {"case-type": "channel", "top": 5, "a0-target": 1.2, "exit-after-mm": 1.5,
                    "waist-growth-sigma": 0.5, "entry-window-mm": 0.75},
    },
    {"name": "triplets", "script": "score_triplets", "options": {"top": 5}},
    {"name": "beamlike_last", "script": "score_beamlike_pairs", "options": {"row-selection": "last", "top": 3}},
    {"name": "beamlike_single", "script": "score_beamlike_pairs", "options": {"score-floor": 2.0}},
    {
        "name": "joint_inner",
        "script": "compare_guiding_beamlike_scores",
        "options": {"triplet-scores-csv": "golden/scoring/triplets/triplet_scores.csv",
                    "beamlike-pair-scores-csv": "golden/scoring/beamlike_last/beamlike_pair_scores.csv"},
    },
    {
        "name": "joint_outer",
        "script": "compare_guiding_beamlike_scores",
        "options": {"triplet-scores-csv": "golden/scoring/triplets/triplet_scores.csv",
                    "beamlike-pair-scores-csv": "golden/scoring/beamlike_single/beamlike_pair_scores.csv",
                    "join-how": "outer", "top": 2},
    },
]


def generate_scoring_goldens(golden: Path) -> None:
    scoring = golden / "scoring"
    shutil.rmtree(scoring, ignore_errors=True)
    scoring.mkdir(parents=True)
    (scoring / "runs.json").write_text(json.dumps(SCORING_RUNS, indent=2) + "\n")
    for run in SCORING_RUNS:
        outdir = f"golden/scoring/{run['name']}"
        arguments = [] if run["script"] == "compare_guiding_beamlike_scores" else list(SCORING_CAMPAIGN)
        for flag, value in run["options"].items():
            arguments += [f"--{flag}", str(value)]
        stdout = run_reference(str(REPO / "scripts" / f"{run['script']}.py"), *arguments, "--outdir", outdir, cwd=DATA)
        (DATA / outdir / "stdout.txt").write_text(stdout)


def generate_particle_goldens(golden: Path) -> None:
    particles = golden / "particles"
    shutil.rmtree(particles, ignore_errors=True)
    particles.mkdir(parents=True)
    (particles / "runs.json").write_text(json.dumps(PARTICLE_RUNS, indent=2) + "\n")

    for run in PARTICLE_RUNS:
        outdir = particles / run["name"]
        stdout = run_reference(
            "scripts/analyze_particle_case.py",
            *particle_run_arguments(run),
            "--outdir", rel(outdir),
            "--overwrite",
        )
        shutil.rmtree(outdir / "plots", ignore_errors=True)
        (outdir / "stdout.txt").write_text(stdout)

    campaign_out = particles / "campaign"
    with tempfile.TemporaryDirectory(prefix="guiding_particle_campaign_") as tmp:
        root = Path(tmp) / "campaign"
        shutil.copytree(DATA / "synthetic" / "campaign", root)
        stdout = run_reference(
            "scripts/analyze_particle_campaign.py",
            "--campaign-root", str(root),
            "--case-glob", "0*",
            check=False,
        )
        for case in sorted(root.iterdir()):
            products = case / "particle_analysis"
            if products.is_dir():
                target = campaign_out / case.name
                target.mkdir(parents=True)
                for csv_path in sorted(products.glob("*.csv")):
                    shutil.copy2(csv_path, target / csv_path.name)
        campaign_out.mkdir(parents=True, exist_ok=True)
        (campaign_out / "stdout.txt").write_text(stdout.replace(str(root), "<CAMPAIGN_ROOT>"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def library_versions() -> dict[str, str]:
    import h5py
    import numpy
    import openpmd_viewer
    import pandas

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "h5py": h5py.__version__,
        "openpmd_viewer": openpmd_viewer.__version__,
    }


def main() -> None:
    campaign = DATA / "synthetic" / "campaign"
    golden = DATA / "golden"
    cases = sorted(p for p in campaign.iterdir() if (p / "diags" / "diag1").is_dir())
    if not cases:
        raise SystemExit(f"no fixtures under {campaign}; run tools/golden/make_synthetic_openpmd.py first")

    for case in cases:
        run_reference(
            "scripts/analyze_case.py",
            "--diag", rel(case / "diags" / "diag1"),
            "--outdir", rel(golden / "cases" / case.name),
            "--overwrite",
            "--no-plots",
        )

    campaign_out = golden / "campaign"
    shutil.rmtree(campaign_out, ignore_errors=True)
    stdout = run_reference(
        "scripts/analyze_campaign.py",
        "--campaign-root", rel(campaign),
        "--outdir", rel(campaign_out),
        "--case-metrics-root", rel(campaign_out / "case_metrics"),
        "--run-cases",
        "--run-triplets",
        "--min-h5", "2",
        "--no-case-plots",
        "--no-triplet-plots",
    )
    (campaign_out / "stdout.txt").write_text(stdout)

    generate_particle_goldens(golden)
    generate_scoring_goldens(golden)

    metadata = {
        "generator": "tools/golden/generate_golden.py",
        "reference_commit": git("rev-parse", "HEAD"),
        "reference_sources_modified": bool(git("status", "--porcelain", "--", "cap_guiding", "scripts")),
        "versions": library_versions(),
        "cases": [case.name for case in cases],
    }
    (golden / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"golden outputs for {len(cases)} cases, the campaign workflow and {len(PARTICLE_RUNS)} particle runs in {golden}")


if __name__ == "__main__":
    main()
