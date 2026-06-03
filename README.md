# guiding_analysis_module

Minimal analysis module for WarpX RZ capillary guiding diagnostics.

Main workflow:

1. Read WarpX RZ openPMD/HDF5 diagnostics from `CASE/diags/diag1`.
2. Generate per-case `guiding_metrics.csv`.
3. Compare channel/uniform/vacuum triplets.
4. Produce lightweight CSV/PNG outputs.

## Single case

```powershell
python .\scripts\analyze_case.py `
  --diag path\to\CASE\diags\diag1 `
  --outdir .\analysis_outputs\case_metrics\CASE_ID `
  --overwrite
```

## Triplet from existing CSVs

```powershell
python .\scripts\compare_triplet.py `
  --channel path\to\channel\guiding_metrics.csv `
  --uniform path\to\uniform\guiding_metrics.csv `
  --vacuum path\to\vacuum\guiding_metrics.csv `
  --outdir .\analysis_outputs\triplets\TRIPLET_ID `
  --overwrite
```

## Triplet from openPMD diagnostics

```powershell
python .\scripts\compare_triplet_cases.py `
  --channel-diag path\to\channel\diags\diag1 `
  --uniform-diag path\to\uniform\diags\diag1 `
  --vacuum-diag path\to\vacuum\diags\diag1 `
  --outdir .\analysis_outputs\triplets\TRIPLET_ID `
  --overwrite-triplet
```