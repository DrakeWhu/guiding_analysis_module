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

After a case becomes reduced-ready, its HDF5 diagnostics may be treated as disposable raw data, provided cleanup safety checks pass.

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

## Cleanup policy

Routine cleanup must only delete explicit HDF5 files from cases that have already been reduced.

A case is safe for HDF5 cleanup only if:

1. `CASE_DIR/guiding_metrics.csv` exists.
2. The CSV is valid and non-empty.
3. The corresponding SLURM task is not running.
4. The deletion list contains only explicit `.h5` paths.
5. Whole case directories are never deleted by routine cleanup.

Example safe cleanup workflow:

```bash
ROOT=/gpfs/home/jrodriguez/warpx_runs/capillaries_bo_full_campaign

squeue -u "$USER" -h -o "%i %T" \
| awk '$2=="RUNNING" && $1 ~ /608072_/ {split($1,a,"_"); print a[2]}' \
| sort -n > "$ROOT/analysis_outputs/running_array_ids.txt"
```

Build the list of reduced, non-running cases:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/gpfs/home/jrodriguez/warpx_runs/capillaries_bo_full_campaign")
analysis = root / "analysis_outputs"

running = {
    line.strip()
    for line in (analysis / "running_array_ids.txt").read_text().splitlines()
    if line.strip()
}

safe = []
skipped_running = []

for csv in sorted(root.glob("*/guiding_metrics.csv")):
    case_dir = csv.parent
    case_id = case_dir.name.split("_", 1)[0]

    try:
        df = pd.read_csv(csv)
        valid = len(df) > 0 and "iteration" in df.columns
    except Exception:
        valid = False

    if not valid:
        continue

    if case_id in running:
        skipped_running.append(case_dir)
    else:
        safe.append(case_dir)

(analysis / "cleanup_safe_not_running.txt").write_text(
    "\n".join(str(p) for p in safe) + ("\n" if safe else "")
)

print("safe cleanup dirs:", len(safe))
print("skipped running dirs:", len(skipped_running))
PY
```

Build an explicit HDF5 deletion list:

```bash
while read -r d; do
    find "$d" -type f -name "*.h5" -print
done < "$ROOT/analysis_outputs/cleanup_safe_not_running.txt" \
> "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt"

wc -l "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt"
```

Estimate size:

```bash
python - <<'PY'
from pathlib import Path

p = Path("/gpfs/home/jrodriguez/warpx_runs/capillaries_bo_full_campaign/analysis_outputs/h5_safe_to_delete_after_metrics.txt")

total = 0
n = 0
for line in p.read_text().splitlines():
    f = Path(line)
    if f.exists():
        total += f.stat().st_size
        n += 1

print("H5 files:", n)
print("Total GiB:", total / 1024**3)
PY
```

Before deletion, inspect the list:

```bash
head -20 "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt"
tail -20 "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt"

grep -v "/capillaries_bo_full_campaign/" "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt" || true
grep -v "\.h5$" "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt" || true
```

If the list is correct, save a manifest and delete only the listed HDF5 files:

```bash
cp "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt" \
   "$ROOT/analysis_outputs/h5_deleted_after_metrics_$(date +%Y%m%d_%H%M%S).txt"

xargs -r rm -v < "$ROOT/analysis_outputs/h5_safe_to_delete_after_metrics.txt"
```

Verify:

```bash
find "$ROOT" -maxdepth 2 -name guiding_metrics.csv | wc -l

while read -r d; do
    find "$d" -type f -name "*.h5" -print
done < "$ROOT/analysis_outputs/cleanup_safe_not_running.txt" | wc -l

du -sh "$ROOT"
```

## C++ tools: guiding_cli and guiding_gui

The Python pipeline above remains the reference. The same reductions are also
available as a C++20 library (`libguiding_core`) with two front ends:

- **`guiding_cli`** — headless, no GL or X11, SLURM friendly. It reproduces
  `analyze_case.py`, `analyze_campaign.py`, `analyze_particle_case.py`,
  `analyze_particle_campaign.py`, `compare_triplet.py` and the `score_*.py`
  scripts, writing the same CSV files.
- **`guiding_gui`** — a Dear ImGui/ImPlot dashboard for browsing a campaign
  while it runs: readiness, case and triplet plots, field maps, particle
  spectra and phase spaces, with background loading and a polling refresh.

The C++ products are byte-identical to the Python ones, with the documented
exceptions in `docs/SPEC.md` (§8). `docs/SPEC.md` is the numerical contract:
openPMD read semantics, the reduction formulas and the CSV text rules.

### Build

```bash
sudo apt install build-essential cmake ninja-build libhdf5-dev libgl-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libwayland-dev libxkbcommon-dev          # GL/X11 packages: only for the GUI

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

Everything except HDF5 (fmt, fast_float, CLI11, Catch2, GLFW, Dear ImGui,
ImPlot, nlohmann/json) is fetched and hash-checked by CMake, or taken from an
installed copy when one is found.

| Option | Default | Purpose |
|---|---|---|
| `GUIDING_BUILD_CLI` / `GUIDING_BUILD_GUI` / `GUIDING_BUILD_TESTS` | ON | Components to build |
| `GUIDING_FETCH_HDF5` | OFF | Build HDF5 1.14 from source instead of using a system/module copy |
| `GUIDING_NATIVE_ARCH` | OFF | `-march=native` |
| `GUIDING_LTO` | OFF | Link-time optimisation |
| `GUIDING_SANITIZE` | "" | e.g. `address;undefined` or `thread` |
| `HDF5_ROOT` | | Where to find HDF5 (an HPC module, or a local build) |

On SUNRISE (GCC 12.1, no GUI):

```bash
module load GCC/12.1.0 CMake HDF5
cmake -S . -B build/hpc -DCMAKE_BUILD_TYPE=Release -DGUIDING_BUILD_GUI=OFF \
      -DHDF5_ROOT="$EBROOTHDF5"
cmake --build build/hpc -j
```

Without an HDF5 module, add `-DGUIDING_FETCH_HDF5=ON`. On macOS use Homebrew
(`brew install cmake hdf5`); on Windows use vcpkg with MSVC 2022. Offline
builds work with `FETCHCONTENT_SOURCE_DIR_<DEP>` or
`FETCHCONTENT_FULLY_DISCONNECTED=ON`.

### Command mapping

| Python script | guiding_cli |
|---|---|
| `analyze_case.py` | `guiding_cli case --diag DIAG --outdir DIR` |
| `analyze_campaign.py` | `guiding_cli campaign --campaign-root ROOT [--run-cases --run-triplets]` |
| `compare_triplet.py` | `guiding_cli triplet --channel A --uniform B --vacuum C` |
| `analyze_particle_case.py` | `guiding_cli particles --diag DIAG --outdir DIR` |
| `analyze_particle_campaign.py` | `guiding_cli particles-campaign --campaign-root ROOT` |
| `score_campaign.py` | `guiding_cli score campaign --campaign-root ROOT` |
| `score_triplets.py` | `guiding_cli score triplets --campaign-root ROOT` |
| `score_beamlike_pairs.py` | `guiding_cli score beamlike-pairs --campaign-root ROOT` |
| `compare_guiding_beamlike_scores.py` | `guiding_cli score joint --campaign-root ROOT` |
| (new) | `guiding_cli inspect --diag DIAG` prints the series layout as JSON |

The flags keep their Python names and the log vocabulary
(`[OK]`, `[SKIP]`, `[FAIL]`, `[MAKE]`, `[USE]`), so existing SUNRISE command
lines carry over. Plot flags are accepted and ignored: the CLI writes no PNGs,
the plots stay with the Python scripts or the GUI. Two additions are
`--threads` (default: `GUIDING_THREADS`, then `SLURM_CPUS_PER_TASK`, then the
CPU affinity mask) and `--no-raw-reads`, which disables the `pread` fast path
used for contiguous datasets.

### Dashboard

```bash
build/release/cpp/gui/guiding_gui --campaign-root ROOT \
    --case-metrics-root analysis_outputs/campaign/case_metrics
```

Panels: campaign browser (readiness chips, scores, filtering, reduce
selected/all), case view (the `plots.py` summary with the plateau window and
the tentative breakdown), triplet view (comparisons, ratios, late window),
overview (score against campaign parameters), fields (intensity map and
lineouts per dump), particles (spectra, acceptance, phase spaces) and a log.
F5 rescans; the campaign is polled every 10 s by default, so a running
simulation shows up without restarting. Any window exports as PNG, and the
plotted series as CSV.

For CI or documentation screenshots there is a headless mode:

```bash
xvfb-run -a build/release/cpp/gui/guiding_gui --campaign-root ROOT \
    --self-test 300 --screenshot shot.png --focus Case
```

### Tests and golden data

`ctest` runs unit tests (numpy/pandas compatibility, CSV text, case-name
grammar, exit selection), parity tests against committed golden CSVs produced
by the Python reference, and I/O tests (the `pread` path equals `H5Dread`,
and thread counts do not change the output). The fixtures live in
`cpp/tests/data` and are regenerated with:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/golden/make_synthetic_openpmd.py   # openPMD fixtures
.venv/bin/python tools/golden/make_scoring_fixtures.py    # CSV-only scoring fixtures
.venv/bin/python tools/golden/generate_golden.py          # run the Python reference
```

### Measured speed-up

One core-i7-4800MQ laptop (8 threads, warm page cache), identical CSV output:

| Workload | Python | guiding_cli |
|---|---|---|
| Field reduction, 12 dumps of 3 × 1600 × 160 (212 MB) | 3.06 s | 0.28 s (1 thread), 0.16 s (8 threads) |
| Particle reduction, one dump of 2 M macroparticles (107 MB) | 10.1 s (read + reduce, no plots) | 5.0 s (including the three CSVs) |

The field path reads each component once (the reference reads `E/r` and `E/t`
twice each) and reduces dumps in parallel; particle reduction is dominated by
the weighted percentiles, which sort each sample once and answer every
percentile from it.

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
    removes transient HDF5 only after valid reduced CSV metrics exist.

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
