#!/usr/bin/env python3
"""Run the Python reference pipeline on the synthetic fixtures.

Outputs land in cpp/tests/data/golden/:
- cases/<case>/: scripts/analyze_case.py (guiding_metrics.csv + single-case score)
- campaign/: scripts/analyze_campaign.py --run-cases --run-triplets (reports,
  case metrics and triplet tables) plus its stdout
metadata.json records the reference commit and library versions.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "cpp" / "tests" / "data"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def run_reference(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    return result.stdout


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

    metadata = {
        "generator": "tools/golden/generate_golden.py",
        "reference_commit": git("rev-parse", "HEAD"),
        "reference_sources_modified": bool(git("status", "--porcelain", "--", "cap_guiding", "scripts")),
        "versions": library_versions(),
        "cases": [case.name for case in cases],
    }
    (golden / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"golden outputs for {len(cases)} cases and the campaign workflow in {golden}")


if __name__ == "__main__":
    main()
