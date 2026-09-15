from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path("scripts/analyze_particle_case.py")


def load_particle_script():
    openpmd = types.ModuleType("openpmd_viewer")
    openpmd.OpenPMDTimeSeries = object
    spec = importlib.util.spec_from_file_location("analyze_particle_case_exit", SCRIPT)
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


class ParticleExitSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = load_particle_script()

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.case_dir = Path(self.tmpdir.name) / "000_case"
        self.case_dir.mkdir()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_resolved(self) -> None:
        (self.case_dir / "resolved_parameters.json").write_text(
            json.dumps(
                {
                    "plasma_start_z": 0.0,
                    "plateau_start_z": 5.0e-3,
                    "plateau_end_z": 10.0e-3,
                    "plasma_end_z": 15.0e-3,
                }
            ),
            encoding="utf-8",
        )

    def test_plateau_exit_uses_resolved_end_not_bare_plateau_length(self) -> None:
        self._write_resolved()
        (self.case_dir / "case.env").write_text(
            'export CAP_PLATEAU_LENGTH_M="5e-3"\n',
            encoding="utf-8",
        )

        target = self.analysis.target_propagation_from_case(
            case_dir=self.case_dir,
            exit_kind="plateau",
            target_propagation_mm=None,
        )

        self.assertEqual(target, 10.0)

    def test_capillary_exit_uses_resolved_plasma_end(self) -> None:
        self._write_resolved()
        target = self.analysis.target_propagation_from_case(
            case_dir=self.case_dir,
            exit_kind="capillary",
            target_propagation_mm=None,
        )
        self.assertEqual(target, 15.0)

    def test_explicit_target_still_takes_precedence(self) -> None:
        self._write_resolved()
        target = self.analysis.target_propagation_from_case(
            case_dir=self.case_dir,
            exit_kind="plateau",
            target_propagation_mm=12.5,
        )
        self.assertEqual(target, 12.5)

    def test_exactly_aligned_exit_selection_is_accepted(self) -> None:
        self.analysis.validate_exit_iteration_alignment(
            {
                "selection_mode": "exit",
                "target_guiding_iteration": 126666,
                "selected_particle_iteration": 126666,
                "target_iteration_delta": 0,
            },
            maximum_target_iteration_delta=0,
        )

    def test_distant_final_dump_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Refusing to analyze a distant"):
            self.analysis.validate_exit_iteration_alignment(
                {
                    "selection_mode": "exit",
                    "target_guiding_iteration": 126666,
                    "selected_particle_iteration": 192000,
                    "target_iteration_delta": 65334,
                },
                maximum_target_iteration_delta=0,
            )

    def test_inconsistent_selection_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent exit selection metadata"):
            self.analysis.validate_exit_iteration_alignment(
                {
                    "selection_mode": "exit",
                    "target_guiding_iteration": 100,
                    "selected_particle_iteration": 101,
                    "target_iteration_delta": 0,
                },
                maximum_target_iteration_delta=1,
            )


if __name__ == "__main__":
    unittest.main()
