from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cap_guiding.diagnostics import (
    resolve_electron_density_field,
    resolve_field_diag_dir,
    resolve_particle_diag_dir,
)


class DiagnosticResolutionTests(unittest.TestCase):
    def test_resolve_field_diag_prefers_fields_over_diag1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "000_f20_chan_n4e18cm3_L10mm_d500um_focm5mm_rz"
            (case / "diags" / "diag1").mkdir(parents=True)
            (case / "diags" / "fields").mkdir(parents=True)

            self.assertEqual(resolve_field_diag_dir(case), case / "diags" / "fields")

    def test_resolve_field_diag_uses_legacy_diag1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "001_f20_uni_n4e18cm3_L10mm_d500um_focm5mm_rz"
            (case / "diags" / "diag1").mkdir(parents=True)

            self.assertEqual(resolve_field_diag_dir(case), case / "diags" / "diag1")

    def test_resolve_field_diag_fails_clearly_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "002_f20_vac_refn4e18cm3_L10mm_d500um_focm5mm_rz"
            with self.assertRaisesRegex(FileNotFoundError, "fields"):
                resolve_field_diag_dir(case)

    def test_resolve_density_legacy(self) -> None:
        self.assertEqual(
            resolve_electron_density_field(["E", "B", "rho_electrons"]),
            "rho_electrons",
        )

    def test_resolve_density_new_defaults_to_plasma_not_sum(self) -> None:
        self.assertEqual(
            resolve_electron_density_field(
                ["E", "B", "rho_plasma_electrons", "rho_ionized_electrons"]
            ),
            "rho_plasma_electrons",
        )

    def test_resolve_density_new_can_select_ionized_explicitly(self) -> None:
        self.assertEqual(
            resolve_electron_density_field(
                ["E", "B", "rho_plasma_electrons", "rho_ionized_electrons"],
                preferred="rho_ionized_electrons",
            ),
            "rho_ionized_electrons",
        )

    def test_resolve_particle_diag_legacy_electrons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            (case / "diags" / "electron_particles" / "openpmd").mkdir(parents=True)

            self.assertEqual(
                resolve_particle_diag_dir(case, "electrons"),
                case / "diags" / "electron_particles" / "openpmd",
            )

    def test_resolve_particle_diag_new_ionized_electrons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            (case / "diags" / "ionized_electrons").mkdir(parents=True)

            self.assertEqual(
                resolve_particle_diag_dir(case, "ionized_electrons"),
                case / "diags" / "ionized_electrons",
            )

    def test_resolve_particle_diag_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            (case / "diags" / "custom_particles").mkdir(parents=True)

            self.assertEqual(
                resolve_particle_diag_dir(
                    case,
                    "electrons",
                    particle_diag_name="custom_particles",
                ),
                case / "diags" / "custom_particles",
            )


if __name__ == "__main__":
    unittest.main()
