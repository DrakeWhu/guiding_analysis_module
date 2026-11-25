from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from laser_guiding.config import bin_edges, load_config, validate_config
from laser_guiding.fields import read_quantity, rolling_mean
from laser_guiding.openpmd_io import open_series, read_field_rz, rz_signed
from laser_guiding.particle_histograms import histograms, kinetic_energy_mev
from laser_guiding.snapshots import Source, completed_step, inventory, select_iterations
from laser_guiding_fixtures import QE, config, fields_file, particles_file


class RZReaderTests(unittest.TestCase):
    def test_real_reader_odd_mode_is_not_mirrored_for_both_axis_orders(self):
        for order in (("z", "r"), ("r", "z")):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as tmp:
                fields_file(Path(tmp) / "fields.h5", order=order)
                ts = open_series(tmp, backend="h5py")
                plane = read_field_rz(ts, 10, "rho_background", m="all", theta=0.0)
                self.assertEqual(plane.arr_rz.shape, (6, 6))
                np.testing.assert_allclose(-plane.arr_rz[plane.r_um > 0] / QE, 1.2e22)
                np.testing.assert_allclose(-plane.arr_rz[plane.r_um < 0] / QE, 0.8e22)
                other = read_field_rz(ts, 10, "rho_background", m="all", theta=np.pi / 2)
                np.testing.assert_allclose(-other.arr_rz[other.r_um > 0] / QE, 1.3e22)
                zero = read_field_rz(ts, 10, "rho_background", m=0)
                np.testing.assert_allclose(-zero.arr_rz / QE, 1e22)
                with self.assertRaisesRegex(ValueError, "Exact field iteration"):
                    read_field_rz(ts, 11, "rho_background")

    def test_transverse_cartesian_conversion_and_density_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            fields_file(Path(tmp) / "fields.h5")
            ts = open_series(tmp, backend="h5py")
            density = read_quantity(ts, 10, config()["fields"][0], 0)
            np.testing.assert_allclose(density.arr_rz[density.r_um > 0], 1.2e16)
            spec = {"kind": "transverse_e2", "m": "all", "unit_scale": 1e-24,
                    "smooth_wavelengths": 2}
            plane = read_quantity(ts, 10, spec, 0, wavelength_m=1.06e-6)
            np.testing.assert_allclose(plane.arr_rz, 4.0)
            self.assertTrue(np.any(plane.r_um < 0))

    def test_legacy_api_and_units_remain_available(self):
        from cap_guiding.openpmd_io import rz_positive as legacy_positive
        info = SimpleNamespace(axes={0: "z", 1: "r"}, z=np.array([1e-6, 2e-6]),
                               r=np.array([-1e-6, 1e-6]))
        arr = np.array([[1., 2.], [3., 4.]])
        result = legacy_positive(arr, info)
        np.testing.assert_array_equal(result.arr_rz, [[2., 4.]])
        np.testing.assert_array_equal(result.z_um, [1., 2.])
        with self.assertRaises(ValueError):
            rz_signed(arr, SimpleNamespace(axes={0: "x", 1: "z"}))

    def test_rolling_mean_preserves_constant_edges(self):
        np.testing.assert_allclose(rolling_mean(np.full((3, 9), 7.), 4), 7.)
        values = np.arange(27.).reshape(3, 9)
        np.testing.assert_array_equal(rolling_mean(values, 1), values)


class ElectronHistogramTests(unittest.TestCase):
    def test_physical_charge_no_extra_cylindrical_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            particles_file(Path(tmp) / "particles.h5")
            ts = open_series(tmp, backend="h5py")
            spec = config()["particles"]
            spectrum, phase, rows = histograms(ts, 10, spec, 0.,
                                               bin_edges(spec["energy_bins"]),
                                               bin_edges(spec["longitudinal_bins"]))
            expected = 14e7 * QE * 1e12
            self.assertAlmostEqual(float(spectrum.sum()), expected)
            self.assertAlmostEqual(float(phase.sum()), expected)
            self.assertAlmostEqual(rows[0]["Q_window_pC"], expected)
            # openPMD-viewer returns u=gamma*beta rather than SI momentum.
            self.assertEqual(rows[0]["selected_macroparticles"], 3)
            spec = copy.deepcopy(spec)
            spec["forward_only"] = True
            spectrum, _, rows = histograms(ts, 10, spec, 0.,
                                           bin_edges(spec["energy_bins"]),
                                           bin_edges(spec["longitudinal_bins"]))
            self.assertAlmostEqual(float(spectrum.sum()), 5e7 * QE * 1e12)

    def test_empty_and_absent_species_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            particles_file(Path(tmp) / "particles.h5", empty=True)
            ts = open_series(tmp, backend="h5py")
            spec = config()["particles"]
            e, z = bin_edges(spec["energy_bins"]), bin_edges(spec["longitudinal_bins"])
            spectrum, _, rows = histograms(ts, 10, spec, 0, e, z)
            self.assertEqual(float(spectrum.sum()), 0.)
            self.assertTrue(rows[0]["registered"])
            self.assertEqual(rows[0]["macroparticles"], 0)
            spec["species"] = ["missing"]
            with self.assertRaisesRegex(ValueError, "absent"):
                histograms(ts, 10, spec, 0, e, z)

    def test_spectral_underflow_overflow_and_phase_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            particles_file(Path(tmp) / "particles.h5")
            ts = open_series(tmp, backend="h5py")
            spec = config()["particles"]
            spectrum, phase, rows = histograms(ts, 10, spec, 0,
                                               np.array([0.1, 0.5]), np.array([-1., 1.]))
            row = rows[0]
            self.assertAlmostEqual(spectrum.sum() + row["Q_energy_underflow_pC"] +
                                   row["Q_energy_overflow_pC"], row["Q_selected_pC"])
            self.assertAlmostEqual(phase.sum() + row["Q_phase_outside_pC"], row["Q_selected_pC"])

    def test_cold_kinetic_energy_retains_precision(self):
        u = np.array([1e-10])
        self.assertGreater(kinetic_energy_mev(u, u * 0, u * 0)[0], 0.)


class SnapshotSelectionTests(unittest.TestCase):
    def test_age_log_bound_exact_selection_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fields_file(folder / "fields.h5")
            self.assertEqual(inventory(folder, min_age_seconds=1e6), {})
            self.assertEqual(inventory(folder, min_age_seconds=0, max_iteration=9), {})
            fields = inventory(folder, min_age_seconds=0, max_iteration=10)
            self.assertEqual(select_iterations(fields, None, [10]), [10])
            with self.assertRaisesRegex(ValueError, "unavailable"):
                select_iterations(fields, None, [11])
            with self.assertRaisesRegex(ValueError, "differ"):
                select_iterations(fields, {}, None)
            fields_file(folder / "duplicate.h5")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                inventory(folder, min_age_seconds=0)

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source"
            path.write_bytes(b"x")
            stat = path.stat()
            source = Source(path, stat.st_size, stat.st_mtime_ns)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                source.check()

    def test_completed_log_ignores_unfinished_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warpx.out"
            path.write_text("STEP 10 ends. Time = 1\nSTEP 11 starts ...\n")
            self.assertEqual(completed_step(path), 10)


class ConfigurationTests(unittest.TestCase):
    def test_resolved_parameters_and_explicit_wavelength(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = config()
            c["parameter_file"] = "resolved_parameters.json"
            c["reference"]["initial_z_m"] = {"parameter": "laser_centroid_initial_z_m"}
            (root / "resolved_parameters.json").write_text(json.dumps({"laser_centroid_initial_z_m": -333e-6}))
            (root / "config.json").write_text(json.dumps(c))
            result = load_config(root / "config.json", root)
            self.assertEqual(result["reference"]["initial_z_m"], -333e-6)

    def test_unknown_keys_and_ambiguous_density_are_rejected(self):
        c = config()
        c["wall_radius"] = 10
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_config(c)
        c = config()
        c["fields"][0]["field"] = "rho"
        with self.assertRaisesRegex(ValueError, "species charge-density"):
            validate_config(c)

    def test_invalid_scales_modes_and_bins(self):
        for key, value in (("m", -1), ("norm", "magic"), ("unit_scale", 0),
                           ("vmin", -1)):
            c = config()
            c["fields"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(c)
        with self.assertRaises(ValueError):
            bin_edges({"min": 0, "max": 1, "count": 2, "spacing": "log"})


if __name__ == "__main__":
    unittest.main()
