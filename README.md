# guiding_analysis_module

Analysis module for WarpX/PyWarpX RZ capillary guiding campaigns.

This module is used in the capillary discharge / guiding simulation pipeline. It does **not** launch WarpX simulations. It analyzes WarpX openPMD/HDF5 diagnostics that already exist on disk, reduces them to lightweight per-case CSV files, and then builds campaign-level and triplet-level products.

The central design principle is:

```text
WarpX openPMD/HDF5 = transient raw data
guiding_metrics.csv = persistent reduced diagnostic
triplets = derived comparison products
```

## Pipeline role

The intended campaign workflow is:

```text
WarpX/PyWarpX input.py
  ↓
SLURM simulation job
  ↓
CASE_DIR/diags/diag1/*.h5
  ↓
guiding_analysis_module
  ↓
CASE_DIR/guiding_metrics.csv
  ↓
optional cleanup of reduced HDF5 files
  ↓
campaign reports / triplet comparisons / BO-ready metrics
```

Each simulation case is the primary data unit. A triplet is only a derived view over three cases:

```text
channel case + uniform baseline + vacuum baseline
```

This means that each case owns its own reduced metrics:

```text
CASE_DIR/guiding_metrics.csv
```

A channel case must not contain copied uniform/vacuum data. The campaign layer is responsible for discovering compatible cases and building triplet products from their local CSVs.

## Case states

A case can be in one of these states:

```text
raw-ready
    The case has enough stable WarpX openPMD/HDF5 diagnostics to compute metrics.

reduced-ready
    The case already has a valid CASE_DIR/guiding_metrics.csv.

usable
    The case is either raw-ready or reduced-ready.
```

A reduced-ready case has usable guiding metrics. This does not establish that
all particle products or future-analysis inputs have been preserved; raw-data
retention requires the separate campaign-level review described below.

## SUNRISE environment

Typical SUNRISE setup:

```bash
module purge
module load Git/2.41.0
module load GCC/12.1.0
module load Python/3.10.12
module load OpenBLAS/0.3.31
module load warpx/26.05-gcc12-openmpi413-all-dims

export MPLBACKEND=Agg

source ~/apps/venvs/guiding-analysis-py310/bin/activate
cd ~/apps/src/guiding_analysis_module
```

The `Python/3.10.12` module must be loaded before activating the `guiding-analysis-py310` virtual environment. Otherwise, the Python executable may fail to find `libpython3.10.so.1.0`.

## Single case analysis

Analyze one WarpX diagnostic directory and write metrics to an output directory:

```bash
python scripts/analyze_case.py \
  --diag /path/to/CASE/diags/diag1 \
  --outdir /path/to/output/CASE_ID \
  --overwrite
```

For campaign operation, the preferred persistent location is inside the case directory:

```text
CASE_DIR/guiding_metrics.csv
```

## Campaign analysis

Set paths:

```bash
ROOT=/gpfs/home/jrodriguez/warpx_runs/capillaries_bo_full_campaign
ANALYZER=/gpfs/home/jrodriguez/apps/src/guiding_analysis_module

cd "$ANALYZER"
```

### CSV-aware dry-run

The dry-run discovers cases, structural triplets, HDF5 readiness, and existing reduced CSV metrics:

```bash
OUTDIR="$ROOT/analysis_outputs/dryrun_csv_aware_$(date +%Y%m%d_%H%M%S)"

python scripts/analyze_campaign.py \
  --campaign-root "$ROOT" \
  --outdir "$OUTDIR" \
  --case-metrics-root "$ROOT" \
  --min-h5 2 \
  --min-last-h5-age-min 10
```

Important summary fields:

```text
cases with valid CSV metrics
    Number of cases already reduced to CASE_DIR/guiding_metrics.csv.

cases usable CSV or HDF5
    Number of cases that either already have valid CSV metrics or have enough stable HDF5 files to be reduced.

triplets CSV complete
    Number of triplets whose channel, uniform and vacuum cases all have valid CSV metrics.

triplets usable CSV or HDF5
    Number of triplets whose three members are usable.
```

### Reduce ready cases to local CSV

Generate `guiding_metrics.csv` for every ready case that does not already have valid metrics:

```bash
OUTDIR="$ROOT/analysis_outputs/case_reduce_$(date +%Y%m%d_%H%M%S)"

python scripts/analyze_campaign.py \
  --campaign-root "$ROOT" \
  --outdir "$OUTDIR" \
  --case-metrics-root "$ROOT" \
  --run-cases \
  --skip-existing \
  --min-h5 2 \
  --min-last-h5-age-min 10 \
  --no-case-plots
```

This writes:

```text
CASE_DIR/guiding_metrics.csv
```

for each successfully reduced case.

Check how many reduced cases exist:

```bash
find "$ROOT" -maxdepth 2 -name guiding_metrics.csv | wc -l
```


## Particle reduced outputs

Particle analysis reads WarpX openPMD/HDF5 particle diagnostics from:

```text
CASE_DIR/diags/plasma_electrons
```

for the current SUNRISE campaign convention. The internal openPMD species name is:

```text
electrons
```

The persistent reduced particle outputs are written under:

```text
CASE_DIR/particle_analysis/
```

The main summary file is:

```text
CASE_DIR/particle_analysis/particle_summary.csv
```

A second reduced file stores accepted-charge curves:

```text
CASE_DIR/particle_analysis/particle_acceptance_curves.csv
```

Each row represents an accepted charge under simultaneous energy and divergence cuts:

```text
Q(E >= E_min_MeV, theta_r <= theta_cut_mrad)
```

Default cuts are:

```text
theta_cut_mrad = 2, 5, 10, 20, 50
E_min_MeV      = 10, 25, 50, 100, 150, 200, 250, 300
```

The intended use is channel-vs-uniform comparison after raw HDF5 cleanup. The file is small enough to keep as a persistent reduced diagnostic and avoids depending on phase-space plots or full spectra.

For vacuum cases or cases without particle diagnostics, the campaign workflow may skip particle analysis. In that situation this file is not required.


## Triplet analysis

Triplets are built from reduced per-case CSV files whenever possible.

```bash
OUTDIR="$ROOT/analysis_outputs/triplet_run_$(date +%Y%m%d_%H%M%S)"

python scripts/analyze_campaign.py \
  --campaign-root "$ROOT" \
  --outdir "$OUTDIR" \
  --case-metrics-root "$ROOT" \
  --run-triplets \
  --skip-existing \
  --min-h5 2 \
  --min-last-h5-age-min 10
```

The triplet layer should not require HDF5 if all three case metrics CSVs already exist.

## Generate case plots from existing CSV metrics

Once a case has been reduced to:

`CASE_DIR/guiding_metrics.csv`

its plots can be generated without reading WarpX/openPMD HDF5 diagnostics again:

```
OUTDIR="$ROOT/analysis_outputs/plot_existing_cases_$(date +%Y%m%d_%H%M%S)"

python scripts/analyze_campaign.py \
--campaign-root "$ROOT" \
--outdir "$OUTDIR" \
--case-metrics-root "$ROOT" \
--plot-existing-cases \
--skip-existing

```

This writes plots to:

`CASE_DIR/plots/`

The sentinel plot is:

`CASE_DIR/plots/guiding_summary_multipanel.png`

Use:

`--overwrite-case-plots`

to regenerate existing plots.

This command is CSV-based. It does not require the original HDF5 files to still exist.


## Shared baselines

The campaign parser supports capillary campaign names such as:

```text
000_f20_chan_n7e17cm3_L5mm_d150um_focm5mm_rz
243_f20_uni_n7e17cm3_L5mm_refd500um_focm5mm_rz
324_f20_vac_L5mm_refd500um_focm5mm_rz
```

Channel cases are distinguished by:

```text
laser case
density
plateau length
diameter
focus offset
```

Uniform and vacuum baselines may be shared across families of channel cases when the campaign keys are compatible.

The output triplet label includes channel diameter to avoid collisions, for example:

```text
f20_n4e18cm3_L5mm_d150um_focm5mm
f20_n4e18cm3_L5mm_d300um_focm5mm
f20_n4e18cm3_L5mm_d500um_focm5mm
```

## Retention and cleanup boundary

This module does not delete raw diagnostics or authorize their removal.
Production cleanup belongs to campaign-workflow's `docs/CLEANUP_SAFETY.md`:
raw and reduced validation evidence, explicit eligibility, a prior file-level
manifest, safe paths, and a separate execute request are required.
A cleanup dry run writes a manifest/evidence; it is not read-only.

The historical manual deletion recipe has been removed from this README.
A non-empty `guiding_metrics.csv` and an inactive scheduler task are not enough
to demonstrate scientific completeness of all retained products. The workflow's
capillary example currently requires only that guiding CSV, while its HDF5
globs can include particle diagnostics. The particle wrapper's non-empty
summary check is not a complete retention contract for acceptance curves,
species-specific products, or downstream scores.

Before enabling production cleanup, the scientific owner must define the
required reduced products per raw diagnostic family and the raw data that
must remain available for future analyses. This documentation does not change
campaign configuration or implement new eligibility checks. See
campaign-workflow's `docs/INTEGRATION_REVIEW.md` for the proposed review agenda.

## Temporal selection and optimizer compatibility

The exact resolved particle-exit path in `cap_guiding/particle_exit.py` requires
the specified particle iteration; a missing exact dump must not silently become
a nearby-frame measurement. Preserve exit-target metadata and the associated
`target_guiding_iteration` in reduced products.

The capillary optimizer uses the corresponding guiding row when that target is
provided; without a usable target it can use a single row or a last-row fallback.
That fallback is not evidence of an exact-exit measurement. Which legacy
observations remain scientifically fit-eligible is still a review decision.

Keep particle summaries, acceptance curves, and species-specific products
separate from field-only metrics. A late-frame beam measurement does not show
the maximum beam achieved during propagation, especially after particle loss.

## Local verification

With the intended scientific environment, run from the repository root:

```bash
PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Use isolated exports and temporary directories when preserving a checkout.
The suite includes synthetic fixtures and mocks; successful tests do not prove
real-diagnostic or HPC validity. The recorded interpreter, dependencies,
warnings, and results are in `docs/BASELINE_2026-10-20.md`.

## Development notes

The module should remain focused on analysis. It should not launch WarpX jobs.

Recommended separation:

```text
simulation layer
    creates case directories and runs WarpX/PyWarpX.

case analysis layer
    converts one case from HDF5/openPMD to CASE_DIR/guiding_metrics.csv.

campaign analysis layer
    discovers cases, detects shared baselines, builds triplets and reports readiness.

cleanup layer
    removes selected HDF5 only after workflow evidence and the reviewed
    scientific retention contract are satisfied.

optimization layer
    consumes CSV metrics and derived scores; it should not depend on raw HDF5.
```

For future BO/MORBO campaigns, the optimizer should consume stable tabular outputs such as:

```text
candidate_id
case_id
parameters
raw physical metrics
derived comparison metrics
score
status
failure_reason
```

Do not make Bayesian optimization depend directly on raw HDF5 dumps.
