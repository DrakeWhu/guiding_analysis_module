from __future__ import annotations

from pathlib import Path

from .metrics import compute_case_rows, write_case_csv
from .plots import save_case_plots


def case_id_from_diag(diag: str | Path) -> str:
    """Infer CASE_ID from CASE/diags/diag1 when possible."""
    diag = Path(diag)

    if diag.name.startswith("diag") and diag.parent.name == "diags":
        return diag.parents[1].name

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

    if make_plots:
        save_case_plots(rows, outdir)

    return csv_path
