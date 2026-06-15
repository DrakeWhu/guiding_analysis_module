from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cap_guiding.campaign import infer_case_info_from_dir


class CampaignFieldDiagnosticResolutionTests(unittest.TestCase):
    def test_infer_case_info_uses_new_fields_diag_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "000_f20_chan_n4e18cm3_L10mm_d500um_focm5mm_rz"
            (case / "diags" / "fields").mkdir(parents=True)
            (case / "diags" / "fields" / "openpmd_000000.h5").touch()

            info = infer_case_info_from_dir(case)

            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.diag_dir, case / "diags" / "fields")
            self.assertEqual(info.h5_count, 1)

    def test_infer_case_info_keeps_legacy_diag1_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "001_f20_uni_n4e18cm3_L10mm_d500um_focm5mm_rz"
            (case / "diags" / "diag1").mkdir(parents=True)
            (case / "diags" / "diag1" / "openpmd_000000.h5").touch()

            info = infer_case_info_from_dir(case)

            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.diag_dir, case / "diags" / "diag1")
            self.assertEqual(info.h5_count, 1)


if __name__ == "__main__":
    unittest.main()
