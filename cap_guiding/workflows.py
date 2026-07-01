from __future__ import annotations

from pathlib import Path

from .metrics import compute_case_rows, write_case_csv
from .plots import save_case_plots
from .singlecase_guiding import ensure_singlecase_guiding_score_csv


_FIELD_DIAG_NAMES = {"fields", "diag1"}


def case_dir_from_diag(diag: str | Path) -> Path:
    """Infer CASE_DIR from a standard field diagnostic path.

    Supported layouts:
      - CASE_DIR/diags/fields  (new ionization-capable campaigns)
      - CASE_DIR/diags/diag1   (legacy campaigns)
    """
    diag = Path(diag)

    if diag.parent.name == "diags" and (
        diag.name in _FIELD_DIAG_NAMES or diag.name.startswith("diag")
    ):
        return diag.parents[1]

    raise ValueError(
        f"Cannot infer CASE_DIR from diagnostic path {diag!s}. "
        "Expected CASE_DIR/diags/fields or CASE_DIR/diags/diagN; "
        "pass --outdir explicitly."
    )


def case_id_from_diag(diag: str | Path) -> str:
    """Infer CASE_ID from a standard field diagnostic path when possible."""
    diag = Path(diag)

    if diag.parent.name == "diags" and (
        diag.name in _FIELD_DIAG_NAMES or diag.name.startswith("diag")
    ):
        return case_dir_from_diag(diag).name

    return diag.name


def ensure_case_metrics(
    diag: str | Path,
    case_metrics_root: str | Path,
    case_id: str | None = None,
    stride: int = 1,
    smooth_um: float = 2.0,
    wake_behind_um: float = 120.0,
    wake_gap_um: float = 5.0,
    lambda0_m: float = 0.8e-6,
    overwrite: bool = False,
    make_plots: bool = True,
    write_singlecase_score: bool = True,
) -> Path:
    """Return guiding_metrics.csv, generating it if missing or overwrite=True."""
    diag = Path(diag)
    case_metrics_root = Path(case_metrics_root)

    if case_id is None:
        case_id = case_id_from_diag(diag)

    outdir = case_metrics_root / case_id
    csv_path = outdir / "guiding_metrics.csv"

    if csv_path.exists() and not overwrite:
        print(f"[USE] existing case metrics: {csv_path}")
        if write_singlecase_score:
            ensure_singlecase_guiding_score_csv(
                csv_path,
                case_id=case_id,
                overwrite=False,
            )
        return csv_path

    print(f"[MAKE] case metrics for {case_id}")
    print(f"       diag   = {diag}")
    print(f"       outdir = {outdir}")

    rows = compute_case_rows(
        diag=diag,
        stride=stride,
        smooth_um=smooth_um,
        wake_behind_um=wake_behind_um,
        wake_gap_um=wake_gap_um,
        lambda0_m=lambda0_m,
    )

    write_case_csv(rows, csv_path)
    print(f"[OK] wrote {csv_path}")

    if write_singlecase_score:
        ensure_singlecase_guiding_score_csv(
            csv_path,
            case_id=case_id,
            overwrite=True,
        )

    if make_plots:
        save_case_plots(rows, outdir)

    return csv_path
