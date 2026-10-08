"""End-to-end checks with actual synthetic openPMD/HDF5 particle files."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from cap_guiding.metrics import C_M_PER_S, E_CHARGE_C, M_E_KG


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_beam_evolution.py"


def write_particles(path: Path, iteration: int, species: str, *, empty: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    n = 0 if empty else 4
    with h5py.File(path / f"particles_{iteration:06d}.h5", "w") as f:
        for key, value in dict(openPMD="1.1.0", basePath="/data/%T/", particlesPath="particles/",
                               meshesPath="meshes/", iterationEncoding="fileBased",
                               iterationFormat="particles_%T.h5").items():
            f.attrs[key] = np.bytes_(value)
        f.attrs["openPMDextension"] = np.uint32(0)
        data = f.create_group(f"data/{iteration}")
        data.attrs.update(time=iteration * 100e-15, dt=100e-15, timeUnitSI=1.)
        particles = data.create_group(f"particles/{species}")
        particles.attrs["particleShape"] = 1.
        for record, dimension in (("position", [1, 0, 0, 0, 0, 0, 0]),
                                   ("positionOffset", [1, 0, 0, 0, 0, 0, 0]),
                                   ("momentum", [1, 1, -1, 0, 0, 0, 0])):
            group = particles.create_group(record)
            group.attrs.update(unitDimension=np.array(dimension, dtype=float), timeOffset=0.)
            for axis in ("x", "y", "z"):
                values = np.zeros(n)
                if record == "position" and axis == "x":
                    values = np.linspace(-2e-6, 2e-6, n)
                if record == "position" and axis == "y":
                    values = np.linspace(1e-6, -1e-6, n)
                if record == "position" and axis == "z":
                    values = np.full(n, (20 + iteration * 10) * 1e-6)
                if record == "momentum" and axis == "z":
                    values = np.full(n, np.sqrt((80 / .51099895 + 1) ** 2 - 1) * M_E_KG * C_M_PER_S)
                dataset = group.create_dataset(axis, data=values)
                dataset.attrs["unitSI"] = 1.
        for record, values, dimension in (
            ("weighting", np.ones(n), [0, 0, 0, 0, 0, 0, 0]),
            ("mass", np.full(n, M_E_KG), [0, 1, 0, 0, 0, 0, 0]),
            ("charge", np.full(n, -E_CHARGE_C), [0, 0, 1, 1, 0, 0, 0]),
        ):
            dataset = particles.create_dataset(record, data=values)
            dataset.attrs.update(unitSI=1., unitDimension=np.array(dimension, dtype=float), timeOffset=0.)


class BeamEvolutionIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.case = Path(self.temp.name) / "003_case"
        self.case.mkdir()
        self.guiding = self.case / "guiding_metrics.csv"
        self.guiding.write_text("iteration,time_fs,z_max_um\n0,0,20\n10,1000,120\n20,2000,220\n")
        self.resolved = self.case / "resolved_parameters.json"
        self.resolved.write_text(json.dumps({
            "plasma_start_z": 20e-6, "plasma_end_z": 220e-6,
            "particle_diagnostic_targets": {"capillary_exit": {"iteration": 20}},
        }))
        self.background = self.case / "diags" / "plasma_electrons"
        self.ionized = self.case / "diags" / "ionized_electrons"
        for it in (0, 10, 20):
            write_particles(self.background, it, "electrons")
            write_particles(self.ionized, it, "ionized_electrons", empty=it == 0)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), "--case-dir", str(self.case),
                               "--species", "electrons,ionized_electrons", "--max-gap-um", "150", *extra],
                              cwd=ROOT, capture_output=True, text=True, timeout=30,
                              env={**os.environ, "MPLBACKEND": "Agg"})

    def read_product(self, name):
        with (self.case / "beam_analysis" / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_real_reader_multispecies_csv_and_legacy_products_preserved(self):
        legacy = self.case / "particle_analysis"
        legacy.mkdir()
        for name in ("particle_summary.csv", "particle_acceptance_curves.csv", "particle_soft50_curves.csv"):
            (legacy / name).write_bytes(b"historical output\n")
        before = {p: p.read_bytes() for p in legacy.iterdir()}
        guiding_before = self.guiding.read_bytes()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        frames = self.read_product("beam_evolution.csv")
        summaries = self.read_product("beam_evolution_summary.csv")
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(summaries), 2)
        empty = next(r for r in frames if r["species_scope"] == "ionized_electrons" and r["iteration"] == "0")
        self.assertEqual(empty["population_status"], "empty")
        self.assertEqual(float(empty["charge_pC"]), 0)
        for row in summaries:
            self.assertEqual(row["integration_status"], "complete")
            self.assertEqual(row["exit_iteration"], "20")
            self.assertAlmostEqual(float(row["charge_exit_snapshot_pC"]), 4 * E_CHARGE_C * 1e12)
        self.assertEqual(before, {p: p.read_bytes() for p in legacy.iterdir()})
        self.assertEqual(guiding_before, self.guiding.read_bytes())

    def test_combined_scope_reduces_concatenated_population(self):
        result = self.run_cli("--combined-scope")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        frames = self.read_product("beam_evolution.csv")
        self.assertEqual(len(frames), 9)
        total = next(r for r in frames if r['species_scope']=='all_electrons' and r['iteration']=='10')
        self.assertAlmostEqual(float(total['charge_pC']), 8 * E_CHARGE_C * 1e12)
        self.assertEqual(total['population_status'], 'ok')

    def test_partial_coverage_option_never_hides_missing_species(self):
        (self.ionized / "particles_000010.h5").unlink()
        result = self.run_cli("--combined-scope", "--allow-partial-coverage")
        self.assertNotEqual(result.returncode, 0)
        total = next(r for r in self.read_product('beam_evolution.csv')
                     if r['species_scope']=='all_electrons' and r['iteration']=='10')
        self.assertEqual(total['population_status'], 'unavailable')

    def test_missing_species_dump_is_unavailable_not_zero(self):
        (self.ionized / "particles_000010.h5").unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        frames = self.read_product("beam_evolution.csv")
        missing = next(r for r in frames if r["species_scope"] == "ionized_electrons" and r["iteration"] == "10")
        self.assertEqual(missing["population_status"], "unavailable")
        self.assertEqual(missing["charge_pC"], "nan")
        summaries = {r["species_scope"]: r for r in self.read_product("beam_evolution_summary.csv")}
        self.assertEqual(summaries["electrons"]["integration_status"], "complete")
        self.assertEqual(summaries["ionized_electrons"]["charge_distance_pC_um"], "nan")

    def test_missing_exact_exit_is_not_replaced(self):
        for path in (self.background, self.ionized):
            (path / "particles_000020.h5").unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        for row in self.read_product("beam_evolution_summary.csv"):
            self.assertEqual(row["exit_snapshot_status"], "unavailable")
            self.assertEqual(row["charge_exit_snapshot_pC"], "nan")

    def test_refuses_overwrite_and_conflicting_plasma_bounds(self):
        first = self.run_cli()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        path = self.case / "beam_analysis" / "beam_evolution.csv"
        before = path.read_bytes()
        second = self.run_cli()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(path.read_bytes(), before)
        conflict = self.run_cli("--overwrite", "--plasma-start-um", "0", "--plasma-end-um", "220")
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("disagree", conflict.stdout)
        self.assertEqual(path.read_bytes(), before)

    def test_campaign_mode_uses_per_case_metadata(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--campaign-root", str(self.case.parent),
                                 "--species", "electrons,ionized_electrons"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.read_product("beam_evolution_summary.csv")), 2)


if __name__ == "__main__":
    unittest.main()
