from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


SCRIPT = Path("scripts/analyze_particle_case.py")


def load_particle_script():
    openpmd = types.ModuleType("openpmd_viewer")
    openpmd.OpenPMDTimeSeries = object
    spec = importlib.util.spec_from_file_location("analyze_particle_case_scopes", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("openpmd_viewer")
    sys.modules["openpmd_viewer"] = openpmd
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("openpmd_viewer", None)
        else:
            sys.modules["openpmd_viewer"] = previous
    return module


class ParticleSpeciesPlotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = load_particle_script()

    def test_combined_plot_retains_legacy_filename(self) -> None:
        self.assertEqual(
            self.analysis.particle_plot_suffix(
                species_scope="all_electrons",
                iteration=42,
                preserve_legacy_name=True,
            ),
            "it00000042",
        )

    def test_separate_species_plot_has_unambiguous_filename(self) -> None:
        self.assertEqual(
            self.analysis.particle_plot_suffix(
                species_scope="nitrogen_ionized_electrons",
                iteration=42,
                preserve_legacy_name=False,
            ),
            "nitrogen_ionized_electrons_it00000042",
        )


if __name__ == "__main__":
    unittest.main()
