from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path("scripts/analyze_particle_case.py")


def load_particle_script():
    openpmd = types.ModuleType("openpmd_viewer")
    openpmd.OpenPMDTimeSeries = object
    spec = importlib.util.spec_from_file_location(
        "analyze_particle_case_exact_exit",
        SCRIPT,
    )
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


class ParticleExactExitSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = load_particle_script()
        from cap_guiding import particle_exit

        cls.particle_exit = particle_exit

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.case_dir = Path(self.tmpdir.name) / "000_case"
        self.case_dir.mkdir()
        self.diag = self.case_dir / "diags" / "plasma_electrons"
        self.diag.mkdir(parents=True)
        self.resolved = self.case_dir / "resolved_parameters.json"
        self.guiding = self.case_dir / "guiding_metrics.csv"
        self._write_resolved()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_resolved(
        self,
        *,
        plateau_iteration: int = 60973,
        capillary_iteration: int = 91459,
    ) -> None:
        self.resolved.write_text(
            json.dumps(
                {
                    "particle_diagnostic_targets": {
                        "plateau_exit": {
                            "target_distance_m": 10.0e-3,
                            "iteration": plateau_iteration,
                            "dump_distance_m": 10.00003e-3,
                            "distance_error_m": 0.00003e-3,
                        },
                        "capillary_exit": {
                            "target_distance_m": 15.0e-3,
                            "iteration": capillary_iteration,
                            "dump_distance_m": 15.00002e-3,
                            "distance_error_m": 0.00002e-3,
                        },
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_guiding(self) -> None:
        self.guiding.write_text(
            "iteration,propagation_mm\n"
            "60000,9.84\n"
            "61000,10.01\n"
            "91000,14.92\n"
            "92000,15.08\n",
            encoding="utf-8",
        )

    def test_reads_plateau_and_capillary_targets_from_resolved_parameters(self) -> None:
        plateau = self.particle_exit.read_resolved_particle_exit_target(
            self.resolved,
            exit_kind="plateau",
        )
        capillary = self.particle_exit.read_resolved_particle_exit_target(
            self.resolved,
            exit_kind="capillary",
        )

        self.assertEqual(plateau["selection_mode"], "exit")
        self.assertEqual(plateau["particle_exit_selection_policy"], "exact_resolved_v1")
        self.assertEqual(plateau["resolved_particle_target_key"], "plateau_exit")
        self.assertEqual(plateau["target_particle_iteration"], 60973)
        self.assertEqual(plateau["target_propagation_mm"], 10.0)
        self.assertEqual(capillary["selection_mode"], "exit")
        self.assertEqual(capillary["particle_exit_selection_policy"], "exact_resolved_v1")
        self.assertEqual(capillary["resolved_particle_target_key"], "capillary_exit")
        self.assertEqual(capillary["target_particle_iteration"], 91459)
        self.assertEqual(capillary["target_propagation_mm"], 15.0)

    def test_exact_particle_iteration_is_accepted(self) -> None:
        with patch.object(self.particle_exit, "open_series", return_value=object()), patch.object(
            self.particle_exit,
            "get_iterations",
            return_value=[60973, 91459],
        ):
            selected = self.particle_exit.require_exact_particle_iteration(
                self.diag,
                target_iteration=60973,
            )

        self.assertEqual(selected["selected_particle_iteration"], 60973)
        self.assertEqual(selected["target_iteration_delta"], 0)

    def test_one_step_neighbour_does_not_replace_missing_exact_exit(self) -> None:
        with patch.object(self.particle_exit, "open_series", return_value=object()), patch.object(
            self.particle_exit,
            "get_iterations",
            return_value=[60972, 91459],
        ):
            with self.assertRaisesRegex(RuntimeError, "Refusing nearest/last fallback"):
                self.particle_exit.require_exact_particle_iteration(
                    self.diag,
                    target_iteration=60973,
                )

    def test_distant_last_dump_does_not_replace_missing_plateau_exit(self) -> None:
        with patch.object(self.particle_exit, "open_series", return_value=object()), patch.object(
            self.particle_exit,
            "get_iterations",
            return_value=[91459],
        ):
            with self.assertRaisesRegex(RuntimeError, "target iteration=60973"):
                self.particle_exit.require_exact_particle_iteration(
                    self.diag,
                    target_iteration=60973,
                )

    def test_exact_exit_can_differ_from_nearest_regular_guiding_frame(self) -> None:
        self._write_guiding()
        with patch.object(self.particle_exit, "open_series", return_value=object()), patch.object(
            self.particle_exit,
            "get_iterations",
            return_value=[60973, 91459],
        ):
            iterations, info = self.analysis.resolve_iterations(
                diag=self.diag,
                which="exit",
                stride=1,
                case_dir=self.case_dir,
                guiding_metrics=self.guiding,
                exit_kind="plateau",
                target_propagation_mm=None,
                downramp_mm=None,
                resolved_parameters=self.resolved,
            )

        self.assertEqual(iterations, [60973])
        self.assertEqual(info["selection_mode"], "exit")
        self.assertEqual(info["particle_exit_selection_policy"], "exact_resolved_v1")
        self.assertEqual(info["selected_particle_iteration"], 60973)
        self.assertEqual(info["target_particle_iteration"], 60973)
        self.assertTrue(info["guiding_context_available"])
        self.assertEqual(info["target_guiding_iteration"], 61000)
        self.assertEqual(info["particle_vs_guiding_iteration_delta"], -27)
        self.assertEqual(info["target_iteration_delta"], 0)

    def test_exact_exit_does_not_require_guiding_metrics(self) -> None:
        with patch.object(self.particle_exit, "open_series", return_value=object()), patch.object(
            self.particle_exit,
            "get_iterations",
            return_value=[60973, 91459],
        ):
            iterations, info = self.analysis.resolve_iterations(
                diag=self.diag,
                which="exit",
                stride=1,
                case_dir=self.case_dir,
                guiding_metrics=None,
                exit_kind="plateau",
                target_propagation_mm=None,
                downramp_mm=None,
                resolved_parameters=self.resolved,
            )

        self.assertEqual(iterations, [60973])
        self.assertEqual(info["selection_mode"], "exit")
        self.assertEqual(info["particle_exit_selection_policy"], "exact_resolved_v1")
        self.assertFalse(info["guiding_context_available"])
        self.assertEqual(info["selected_particle_iteration"], 60973)

    def test_explicit_propagation_override_is_rejected_in_authoritative_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.analysis.resolve_iterations(
                diag=self.diag,
                which="exit",
                stride=1,
                case_dir=self.case_dir,
                guiding_metrics=None,
                exit_kind="plateau",
                target_propagation_mm=10.0,
                downramp_mm=None,
                resolved_parameters=self.resolved,
            )

    def test_missing_target_definition_fails_closed(self) -> None:
        self.resolved.write_text(
            json.dumps({"particle_diagnostic_targets": {}}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "lack particle target 'plateau_exit'"):
            self.particle_exit.read_resolved_particle_exit_target(
                self.resolved,
                exit_kind="plateau",
            )


if __name__ == "__main__":
    unittest.main()
