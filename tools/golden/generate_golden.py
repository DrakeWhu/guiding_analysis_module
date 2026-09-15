#!/usr/bin/env python3
"""Run the Python reference pipeline on the synthetic fixtures.

Outputs land in cpp/tests/data/golden/ together with metadata.json, which
records the reference commit and library versions the goldens came from.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "cpp" / "tests" / "data"


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
        outdir = golden / "cases" / case.name
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_case.py",
                "--diag",
                str((case / "diags" / "diag1").relative_to(REPO)),
                "--outdir",
                str(outdir.relative_to(REPO)),
                "--overwrite",
                "--no-plots",
            ],
            cwd=REPO,
            check=True,
        )

    metadata = {
        "generator": "tools/golden/generate_golden.py",
        "reference_commit": git("rev-parse", "HEAD"),
        "reference_sources_modified": bool(git("status", "--porcelain", "--", "cap_guiding", "scripts")),
        "versions": library_versions(),
        "cases": [case.name for case in cases],
    }
    (golden / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"golden outputs for {len(cases)} cases in {golden}")


if __name__ == "__main__":
    main()
