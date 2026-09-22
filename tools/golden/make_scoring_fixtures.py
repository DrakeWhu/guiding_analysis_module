#!/usr/bin/env python3
"""Write a CSV-only campaign for the scoring-layer parity tests.

The scoring scripts (score_campaign, score_triplets, score_beamlike_pairs,
compare_guiding_beamlike_scores) consume reduced CSVs only, so this fixture
contains empty case directories (discovered by name) plus
case_metrics/<case>/guiding_metrics.csv and particle_analysis/particle_summary.csv
with deterministic profiles and the edge cases the scorers branch on:

- NaN a0 rows (valid_fraction < 1) and a case whose NaNs break the score
- a vacuum sampled at other iterations (row-wise reference falls back to medians)
- a missing CSV, an empty CSV, a CSV without a0_peak
- a channel/uniform pair without vacuum (incomplete triplet)
- particle summaries in the old format (no beamlike columns), with several
  rows, without a required column, or missing
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cap_guiding.beamlike import add_beamlike_metrics  # noqa: E402

DEFAULT_OUT = REPO / "cpp" / "tests" / "data" / "synthetic" / "scoring"


def jitter(*keys: float) -> float:
    """Deterministic pseudo-noise in [-1, 1]."""
    seed = 0.0
    for i, key in enumerate(keys):
        seed += (i + 1) * 12.9898 * key
    value = math.sin(seed) * 43758.5453
    return 2.0 * (value - math.floor(value)) - 1.0


def case_name(index: int, kind: str, plateau_mm: int, focus: str, diameter: int) -> str:
    if kind == "chan":
        return f"{index:03d}_f20_chan_n4e18cm3_L{plateau_mm}mm_d{diameter}um_foc{focus}_rz"
    if kind == "uni":
        return f"{index:03d}_f20_uni_n4e18cm3_L{plateau_mm}mm_foc{focus}_rz"
    return f"{index:03d}_f20_vac_L{plateau_mm}mm_foc{focus}_rz"


def metrics_rows(kind: str, plateau_mm: int, diameter: int, focus: str, *, iteration_offset: int = 0,
                 nan_every: int = 0) -> list[dict[str, float]]:
    rows = []
    focus_shift = -0.2 if focus.startswith("m") else 0.0
    for k in range(41):
        iteration = 250 * k + iteration_offset
        z = iteration * 1.0e-3
        inside = 5.0 <= z <= 5.0 + plateau_mm
        if kind == "chan":
            guide = 1.0 + 0.35 * math.exp(-(((diameter - 150) / 60.0) ** 2))
            a0 = 1.05 * guide * math.exp(-max(0.0, z - 5.0 - plateau_mm) / 3.0) if z >= 5.0 else 0.9 + 0.03 * z
            waist = 10.0 * (1.0 + (0.02 if inside else 0.08) * max(0.0, z - 5.0)) * (1.0 + (diameter - 150) / 900.0)
        elif kind == "uni" and plateau_mm == 3:
            # a strong uniform plasma that beats the channels (negative references)
            a0 = (0.9 + 0.03 * z) if z < 5.0 else 1.5 * math.exp(-(z - 5.0) / 40.0)
            waist = 10.0 * (1.0 + 0.03 * max(0.0, z - 5.0))
        elif kind == "uni":
            a0 = (0.9 + 0.03 * z) if z < 5.0 else 1.05 * math.exp(-(z - 5.0) / 4.0)
            waist = 10.0 * (1.0 + 0.05 * max(0.0, z - 5.0))
        else:
            a0 = 0.9 * math.exp(-z / 12.0)
            waist = 10.0 * math.sqrt(1.0 + (z / 6.0) ** 2)
        a0 *= 1.0 + 0.02 * jitter(k, plateau_mm, diameter, len(kind)) + focus_shift * 0.1
        waist *= 1.0 + 0.01 * jitter(plateau_mm, k, diameter, 7.0)
        if nan_every and k % nan_every == 3:
            a0 = float("nan")
        rows.append(
            {
                "iteration": iteration,
                "propagation_mm": iteration * 1.0e-3,
                "a0_peak": a0,
                "waist_um": waist,
                "peak_I_proxy": a0 * a0 * 1.0e24,
                "energy_proxy": 1.0e9 * a0 * a0 * waist * waist / 100.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(kind: str, plateau_mm: int, diameter: int, iteration: int, *, transverse: bool) -> dict[str, object]:
    uniform_strength = 1.33 if plateau_mm == 3 else 0.8  # L3: neutral against d150, negative otherwise
    strength = {"chan": 1.0 + 0.4 * math.exp(-(((diameter - 150) / 60.0) ** 2)), "uni": uniform_strength,
                "vac": 0.1}[kind]
    j = jitter(plateau_mm, diameter, iteration, len(kind))
    row: dict[str, object] = {
        "species_scope": "electrons",
        "iteration": iteration,
        "n_macroparticles_hot": int(900 * strength * (1.0 + 0.1 * j)),
        "charge_hot_pC": 420.0 * strength * (1.0 + 0.05 * j) * plateau_mm / 2.0,
        "E95_hot_MeV": 140.0 * strength * (1.0 + 0.03 * j),
        "Emean_hot_MeV": 95.0 * strength,
        "Emax_hot_MeV": 260.0 * strength * (1.0 + 0.02 * j),
        "q_long_min_hot_mm": 6.8,
        "q_long_max_hot_mm": 6.8 + 0.15 / strength,
    }
    if transverse:
        row.update(
            {
                "transverse_status": "ok",
                "n_macroparticles_transverse": row["n_macroparticles_hot"],
                "weight_transverse": 2.6e9 * strength,
                "theta_x_rms_mrad": 3.0 / strength + 0.1 * j,
                "theta_y_rms_mrad": 2.5 / strength,
                "theta_rms_mrad": math.hypot(3.0 / strength + 0.1 * j, 2.5 / strength),
                "theta_x_p95_mrad": 6.0 / strength,
                "theta_y_p95_mrad": 5.0 / strength,
                "theta_r_p95_mrad": 7.5 / strength,
                "x_rms_um": 3.1,
                "y_rms_um": 2.9,
                "x_p95_um": 6.2,
                "y_p95_um": 5.8,
                "emit_x_norm_mm_mrad": 1.4 / strength,
                "emit_y_norm_mm_mrad": 1.2 / strength,
                "emit_geom_norm_mm_mrad": math.sqrt(1.4 * 1.2) / strength,
                "transverse_theta_rms_component": 1.0 / (1.0 + 3.9 / strength / 5.0),
                "transverse_theta_p95_component": 1.0 / (1.0 + 7.5 / strength / 10.0),
                "transverse_emit_component": 1.0 / (1.0 + 1.3 / strength / 2.0),
                "beam_transverse_quality_score": 1000.0 * strength / (1.0 + 1.0 / strength),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    root = args.outdir
    shutil.rmtree(root, ignore_errors=True)
    campaign = root / "campaign"
    metrics_root = root / "case_metrics"

    index = 0
    cases: list[tuple[str, str, int, str, int]] = []
    for plateau_mm in (2, 3):
        for focus in ("0um", "m200um"):
            for diameter in (100, 150, 200):
                cases.append((case_name(index, "chan", plateau_mm, focus, diameter), "chan", plateau_mm, focus, diameter))
                index += 1
            cases.append((case_name(index, "uni", plateau_mm, focus, 0), "uni", plateau_mm, focus, 0))
            index += 1
            cases.append((case_name(index, "vac", plateau_mm, focus, 0), "vac", plateau_mm, focus, 0))
            index += 1
    # channel/uniform pair without a vacuum reference
    cases.append((case_name(index, "chan", 4, "0um", 150), "chan", 4, "0um", 150))
    index += 1
    cases.append((case_name(index, "uni", 4, "0um", 0), "uni", 4, "0um", 0))

    for name, kind, plateau_mm, focus, diameter in cases:
        (campaign / name).mkdir(parents=True, exist_ok=True)
        csv_path = metrics_root / name / "guiding_metrics.csv"
        summary_path = metrics_root / name / "particle_analysis" / "particle_summary.csv"

        label = (kind, plateau_mm, focus, diameter)
        if label == ("chan", 3, "m200um", 100):
            continue  # no reduced products at all
        if label == ("chan", 2, "m200um", 200):
            write_csv(csv_path, [], fieldnames=["iteration", "propagation_mm", "a0_peak", "waist_um"])
        elif label == ("uni", 3, "0um", 0):
            rows = metrics_rows(kind, plateau_mm, diameter, focus)
            write_csv(csv_path, [{k: v for k, v in row.items() if k != "a0_peak"} for row in rows])
        else:
            offset = 50 if label == ("vac", 2, "m200um", 0) else 0
            nan_every = 4 if label == ("chan", 3, "0um", 200) else (1 if label == ("chan", 2, "0um", 200) else 0)
            write_csv(csv_path, metrics_rows(kind, plateau_mm, diameter, focus, iteration_offset=offset,
                                             nan_every=nan_every))

        if kind == "vac" or label == ("uni", 2, "m200um", 0):
            continue  # vacuum has no particles; one uniform summary is missing
        transverse = not (plateau_mm == 3 and focus == "m200um")
        rows = [summary_row(kind, plateau_mm, diameter, 9000, transverse=transverse)]
        if label == ("chan", 2, "0um", 150):
            rows.insert(0, summary_row(kind, plateau_mm, diameter, 7000, transverse=transverse))
        if label == ("chan", 3, "0um", 150):
            del rows[0]["E95_hot_MeV"]
        new_format = not (plateau_mm == 3 and focus == "m200um")
        if new_format:
            rows = [add_beamlike_metrics(row) for row in rows]
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(summary_path, rows, fieldnames=fieldnames)

    print(f"wrote {len(cases)} scoring cases to {root}")


if __name__ == "__main__":
    main()
