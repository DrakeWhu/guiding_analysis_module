from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np

from laser_guiding.config import bin_edges, validate_config
from laser_guiding.particle_histograms import QE, ME_C2_MEV, histograms
from laser_guiding.visualization import _empty_transverse, _field_figure, visualize
from laser_guiding_fixtures import C, ME, config, make_case


def transverse_config():
    return {"position_bins_um": {"min": -10, "max": 10, "count": 20},
            "momentum_bins_mc": {"min": -10, "max": 10, "count": 20}}


class HotElectronTests(unittest.TestCase):
    def test_charge_threshold_is_independent_of_display_selection_and_bins(self):
        energy = np.array([0., 20., 30., 40.])
        ux = np.array([0., 1., -2., 3.])
        uy = -ux
        uz = np.sqrt((1 + energy / ME_C2_MEV) ** 2 - 1 - ux ** 2 - uy ** 2)
        uz[2] *= -1
        arrays = {"w": np.array([100., 2., 3., 5.]) * 1e7,
                  "x": np.array([0., 1., 2., 200.]) * 1e-6,
                  "y": np.array([0., -1., -2., -200.]) * 1e-6,
                  "z": np.zeros(4), "ux": ux, "uy": uy, "uz": uz}
        calls = []
        def get_particle(var_list, **kwargs):
            calls.append(var_list)
            return [arrays[name] for name in var_list]
        ts = SimpleNamespace(iterations=[10], avail_species=["background"], get_particle=get_particle)
        spec = config()["particles"]
        spec.update(energy_min_mev=25., forward_only=True,
                    transverse_phase_space=transverse_config())
        transverse = _empty_transverse(spec)
        spectrum, phase, rows = histograms(ts, 10, spec, 0.,
                                           np.array([10., 35.]), np.array([-1., 1.]),
                                           transverse=transverse)
        row = rows[0]
        unit = 1e7 * QE * 1e12
        self.assertAlmostEqual(row["Q_ge_10_MeV_pC"], 10 * unit)
        self.assertAlmostEqual(row["Q_selected_pC"], 5 * unit)
        self.assertAlmostEqual(row["Q_energy_overflow_pC"], 5 * unit)
        self.assertEqual(spectrum.sum(), 0.)
        self.assertEqual(phase.sum(), 0.)
        for axis in ("x", "y"):
            self.assertAlmostEqual(
                transverse[f"charge_{axis}_u{axis}_pc"].sum() + row[f"Q_{axis}_u{axis}_outside_pC"],
                row["Q_selected_pC"])
        self.assertEqual(calls, [["w"], ["z", "ux", "uy", "uz", "x", "y"]])

    def test_full_domain_axes_fill_a_balanced_horizontal_layout(self):
        c = config()
        c["fields"] *= 3
        view = dict(c["views"][0], transverse_um=[-300., 300.], longitudinal_um=[-570., 230.])
        fig, axes, color_axes = _field_figure(c, view, {})
        try:
            for ax in axes:
                ax.set(xlim=view["longitudinal_um"], ylim=view["transverse_um"], aspect="equal")
            fig.canvas.draw()
            self.assertEqual(len(color_axes), 3)
            centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axes]
            self.assertLess(centers[0], 0.3)
            self.assertGreater(centers[-1], 0.7)
            for ax in axes:
                box = ax.get_position()
                physical_ratio = box.height * fig.get_figheight() / (box.width * fig.get_figwidth())
                self.assertAlmostEqual(physical_ratio, 0.75)
        finally:
            plt.close(fig)

    def test_real_openpmd_transverse_outputs_charge_csv_and_movies(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            for path in (case / "diags/plasma_electrons").glob("*.h5"):
                with h5py.File(path, "r+") as handle:
                    iteration = next(iter(handle["data"]))
                    species = handle[f"data/{iteration}/particles/background"]
                    species["momentum/z"][:] = 50. * ME * C
                    species["position/x"][:] = np.array([-1., 1., 30.]) * 1e-6
            c = config()
            c["particles"].update(energy_min_mev=10., transverse_phase_space=transverse_config())
            c["particles"]["energy_bins"] = {"min": 10., "max": 100., "count": 20, "spacing": "log"}
            validate_config(c)
            out = Path(tmp) / "out"
            metadata = visualize(case, c, out, requested=[10, 20], with_particles=True,
                                 static_diagnostics=True, min_age_seconds=0, movie=True)
            expected = 14e7 * QE * 1e12
            self.assertAlmostEqual(metadata["frames"][0]["Q_ge_10_MeV_pC"], expected)
            with (out / "electron_charge.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["Q_ge_10_MeV_pC"]), expected)
            with np.load(out / "particle_histograms/00000010.npz") as saved:
                self.assertAlmostEqual(saved["charge_x_ux_pc"].sum(), 7e7 * QE * 1e12)
                self.assertAlmostEqual(saved["charge_y_uy_pc"].sum(), expected)
            row = metadata["frames"][0]["electron_species"][0]
            self.assertAlmostEqual(row["Q_x_ux_outside_pC"], 7e7 * QE * 1e12)
            self.assertEqual(len(list((out / "particle_frames").glob("electrons_transverse_*.png"))), 2)
            import imageio.v2 as imageio
            with imageio.get_reader(str(out / "electrons_transverse.mp4")) as reader:
                self.assertEqual(reader.count_frames(), 2)

    def test_empty_hot_population_still_writes_zero_charge_and_transverse_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            c = config()
            c["particles"].update(energy_min_mev=10., transverse_phase_space=transverse_config())
            c["particles"]["energy_bins"] = {"min": 10., "max": 100., "count": 20, "spacing": "log"}
            metadata = visualize(case, c, Path(tmp) / "out", requested=[10], with_particles=True,
                                 static_diagnostics=True, min_age_seconds=0)
            self.assertEqual(metadata["frames"][0]["Q_ge_10_MeV_pC"], 0.)
            self.assertEqual(metadata["frames"][0]["Q_selected_pC"], 0.)

    def test_transverse_log_bins_are_rejected(self):
        c = config()
        c["particles"]["transverse_phase_space"] = transverse_config()
        c["particles"]["transverse_phase_space"]["momentum_bins_mc"].update(min=1., spacing="log")
        with self.assertRaisesRegex(ValueError, "must be linear"):
            validate_config(c)


if __name__ == "__main__":
    unittest.main()
