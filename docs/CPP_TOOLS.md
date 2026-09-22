# guiding_cli and guiding_gui

A practical guide to the C++ tools: how to install them, how to run them, where
things live, and how the parity with the Python pipeline is checked.

The Python package stays the reference implementation. The C++ tools read the
same diagnostics and write the same CSV files, so they can be used wherever the
scripts are used today, and the two can be compared at any time.

- [Quick start](#quick-start)
- [Install](#install)
- [Run](#run)
- [The dashboard](#the-dashboard)
- [Repository map](#repository-map)
- [How a case flows through the code](#how-a-case-flows-through-the-code)
- [Parity and tests](#parity-and-tests)
- [Checking against your own data](#checking-against-your-own-data)
- [Performance](#performance)
- [Extending](#extending)
- [Troubleshooting](#troubleshooting)

## Quick start

```bash
sudo apt install build-essential cmake ninja-build libhdf5-dev     # once
tools/build.sh                                                     # configure + build + test
build/release/cpp/cli/guiding_cli campaign --campaign-root /path/to/campaign
```

`tools/build.sh` takes a preset (`release` by default, also `hpc`, `dev`,
`tsan`) and passes `HDF5_ROOT` / `GUIDING_FETCH_HDF5` through, so on a cluster:

```bash
module load GCC/12.1.0 CMake HDF5
HDF5_ROOT="$EBROOTHDF5" tools/build.sh hpc
```

The equivalent plain CMake is:

```bash
cmake --preset release          # or: cmake -S . -B build/release -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build --preset release -j
ctest --preset release
```

## Install

Only HDF5 has to exist on the machine. fmt, fast_float, CLI11, Catch2, GLFW,
Dear ImGui, ImPlot and nlohmann/json are downloaded by CMake with pinned
versions and SHA-256 hashes, or taken from an installed copy when one is found.

| System | Command |
|---|---|
| Debian / Ubuntu | `sudo apt install build-essential cmake ninja-build libhdf5-dev` |
| Debian / Ubuntu, GUI too | add `libgl-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libwayland-dev libxkbcommon-dev` |
| macOS | `brew install cmake hdf5` |
| HPC module | `module load GCC/12.1.0 CMake HDF5` then `-DHDF5_ROOT="$EBROOTHDF5"` |
| No HDF5 anywhere | `-DGUIDING_FETCH_HDF5=ON` (builds HDF5 1.14.6 from source, a few minutes) |
| Windows | vcpkg + MSVC 2022 (`vcpkg install hdf5`) |

Build options:

| Option | Default | Purpose |
|---|---|---|
| `GUIDING_BUILD_CLI` | ON | The headless binary |
| `GUIDING_BUILD_GUI` | ON | The dashboard; turn OFF on compute nodes (no GL/X11 needed then) |
| `GUIDING_BUILD_TESTS` | ON | Catch2 suite |
| `GUIDING_FETCH_HDF5` | OFF | Build HDF5 from source instead of finding one |
| `GUIDING_NATIVE_ARCH` | OFF | `-march=native` |
| `GUIDING_LTO` | OFF | Link-time optimisation |
| `GUIDING_SANITIZE` | "" | `address;undefined` or `thread` |

Offline machines: pre-download the dependencies on a connected machine and
point at them with `FETCHCONTENT_SOURCE_DIR_<NAME>` (`FMT`, `FASTFLOAT`,
`CLI11`, `CATCH2`, `GLFW3`, `IMGUI`, `IMPLOT`, `NLOHMANN_JSON`), or set
`FETCHCONTENT_FULLY_DISCONNECTED=ON` when the caches already exist.

The binaries are self-contained: copy `guiding_cli` to the cluster and it runs,
as long as the HDF5 it was linked against is available there.

## Run

Every subcommand mirrors one script, keeps its flag names, and writes the same
files. `--help` works on each one.

| Python | guiding_cli |
|---|---|
| `analyze_case.py` | `case --diag DIAG --outdir DIR` |
| `analyze_campaign.py` | `campaign --campaign-root ROOT [--run-cases --run-triplets]` |
| `compare_triplet.py` | `triplet --channel A --uniform B --vacuum C` |
| `analyze_particle_case.py` | `particles --diag DIAG --outdir DIR` |
| `analyze_particle_campaign.py` | `particles-campaign --campaign-root ROOT` |
| `score_campaign.py` | `score campaign --campaign-root ROOT` |
| `score_triplets.py` | `score triplets --campaign-root ROOT` |
| `score_beamlike_pairs.py` | `score beamlike-pairs --campaign-root ROOT` |
| `compare_guiding_beamlike_scores.py` | `score joint --campaign-root ROOT` |
| — | `inspect --diag DIAG` prints iterations, meshes, species, dtypes and whether the fast read path applies |

Examples:

```bash
# one case
guiding_cli case --diag CASE/diags/diag1 --outdir analysis_outputs/case_metrics/CASE_ID

# campaign dry run (readiness reports only), then the real thing
guiding_cli campaign --campaign-root /scratch/campaign --outdir analysis_outputs/campaign
guiding_cli campaign --campaign-root /scratch/campaign --outdir analysis_outputs/campaign \
                     --run-cases --run-triplets --skip-existing

# particles at the plateau exit, using the exact iteration recorded by the input
guiding_cli particles --diag CASE/diags/plasma_electrons --outdir CASE/particle_analysis \
                      --which exit --resolved-parameters CASE/resolved_parameters.json \
                      --species electrons,ionized_electrons

# scores
guiding_cli score campaign --campaign-root /scratch/campaign --case-type channel
guiding_cli score triplets --campaign-root /scratch/campaign
guiding_cli score beamlike-pairs --campaign-root /scratch/campaign --row-selection last
guiding_cli score joint --campaign-root /scratch/campaign
```

Differences from the scripts, by design:

- **No PNGs.** Plot flags (`--bins`, `--spectrum-log-y`, …) are accepted and
  ignored so command lines carry over; use the GUI or the Python plotting.
- **`--threads`** defaults to `GUIDING_THREADS`, then `SLURM_CPUS_PER_TASK`,
  then the CPU affinity mask. `--no-raw-reads` disables the `pread` fast path
  (contiguous, unfiltered, little-endian datasets are read without going
  through HDF5, which parallelises well on GPFS/Lustre).
- **`particles-campaign` runs in one process** instead of one subprocess per
  case, with per-case logs kept in order.

In a batch script:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=32
module load GCC/12.1.0 HDF5
srun guiding_cli campaign --campaign-root "$CAMPAIGN" --outdir "$CAMPAIGN/analysis_outputs/campaign" \
                          --run-cases --run-triplets --skip-existing --min-h5 2
```

## The dashboard

```bash
guiding_gui --campaign-root ROOT --case-metrics-root ROOT/analysis_outputs/campaign/case_metrics
```

| Panel | What it shows |
|---|---|
| Campaign | Cases with h5 count, newest-dump age, raw/reduced chips, single-case score; triplets with member status. Sort, filter, Ctrl+click to mark, then **Reduce marked** / **Reduce all ready** |
| Case | The `plots.py` summary: front margin, waist, a0 (or peak I) normalised, energy, \|Ez\| in GV/m, z(Ez max) − z_peak, with the plateau window and the tentative breakdown marker; the single-case score sidecar; the metrics table |
| Triplet | Channel/uniform/vacuum comparisons, ratio plots with the y = 1 line and the late window, late summary and ratio tables, the wide table |
| Overview | Single-case score against plateau, density, diameter, focus or f-number; click a point to open that case |
| Fields | Intensity map of one dump (linear or log), I_z with the smoothing window, radial profile with the waist, on-axis E_z with the wake window |
| Particles | Spectra (all and hot), the acceptance grid and Q(E_min) curves, phase spaces, and the full summary row |
| Log | Background jobs and messages |

Useful details: **F5** rescans; the campaign is polled every 10 s so a running
simulation appears without restarting; panels are dockable and the layout is
saved in `~/.config/guiding_gui/imgui.ini`; **PNG** and **CSV** buttons export
the window and the plotted series to `--export-dir` (default
`./guiding_gui_exports`).

Headless, for CI or screenshots:

```bash
xvfb-run -a guiding_gui --campaign-root ROOT --self-test 300 --screenshot shot.png \
                        --focus Case --tab Summary
```

`--self-test N` renders N frames while cycling through cases and triplets,
prints frame-time statistics, and exits non-zero if anything logged an error.

## Repository map

```text
cpp/core/            libguiding_core: everything that computes
  io/                HDF5 RAII, openPMD series discovery, thetaMode slices, particle reader
  numeric/           numpy-compatible pieces (pairwise sums, median, convolve, round-half-even)
  table/             CSV reader/writer, Record (dict-like row), Frame (DataFrame-like), Python float text
  physics/           field metrics, particle summary, transverse, beamlike, soft50
  campaign/          case-name grammar, discovery, readiness, triplets, reports
  products/          what the scripts produce: case_reduction, particle_reduction, particle_campaign,
                     triplet_tables, singlecase_score, scoring, beamlike_pairs, joint_scores,
                     field_map and particle_view (for the GUI)
  exec/              thread count and parallel_for
cpp/cli/             one file per subcommand, argument parsing only
cpp/gui/             data_store (all I/O, caching, jobs) + one file per panel
cpp/tests/           unit, parity and I/O tests; data/ holds fixtures and goldens
tools/golden/        fixture generators and the golden runner (Python)
docs/SPEC.md         the numerical contract that was reverse-engineered
```

If you want to change… | open…
---|---
a field metric | `cpp/core/src/physics/field_metrics.cpp`
what a particle summary row contains | `cpp/core/src/physics/particles.cpp`
how cases are discovered or judged ready | `cpp/core/src/campaign/discovery.cpp`
a CLI flag | the matching `cpp/cli/*_command.cpp`
a plot or panel | `cpp/gui/*_panel.cpp`
anything about file loading in the GUI | `cpp/gui/data_store.cpp`

## How a case flows through the code

```text
CASE/diags/diag1/*.h5
  io::FileSeries::scan                 iterations from the /data groups of every file
  io::read_thetamode_component         E/r, E/t, E/z (all modes, unitSI applied)
  io::cartesian_xy, component_slice    theta = 0 slice, positive-r half
  physics::weighted_laser_metrics      z_peak, waist, peak I, a0, energy proxy
  physics::wake_metrics                Ez max/min/absmax/rms and their positions
  products::compute_case_rows          the above per iteration, in parallel
  products::write_guiding_metrics_csv  guiding_metrics.csv (csv.DictWriter text)
  products::ensure_singlecase_guiding_score_csv
                                       guiding_singlecase_score.csv sidecar
```

Particles follow the same shape: `io::read_particle_dump` (openpmd-viewer
semantics, including the ED-PIC weighting correction) →
`physics::summarize_dump` / `summarize_acceptance_curves` /
`summarize_soft50_curve` → `products::run_particle_case` writes the three CSVs.
The scoring layers read CSVs and write CSVs; they never touch HDF5.

## Parity and tests

`ctest` runs 38 cases in under a second:

- **Unit** — numpy behaviours (pairwise summation, median, convolution,
  round-half-even, `argmax` with NaN), pandas number parsing, Python `repr`
  floats and CSV quoting, the case-name grammar, `case.env` parsing with
  `shlex` semantics, exit-iteration selection, globbing, and the GUI data store.
- **Parity** — every product compared against committed goldens that were
  generated by running the Python scripts: `guiding_metrics.csv`, the
  single-case score, campaign reports, triplet tables, the three particle CSVs
  (9 runs covering `last`, `all`, both `exit` modes, multi-species, custom
  cuts), the particle campaign including its stdout, and 7 scoring runs.
- **I/O** — the `pread` fast path returns exactly what `H5Dread` returns, and
  `--threads 1` and `--threads 8` produce identical bytes.

Results are byte-identical, except two documented cases that use a 1e-12
tolerance (`docs/SPEC.md` §8): numpy's vectorized `tanh` differs from libm by
1 ULP in the triplet `reference_factor`, and the joint correlations go through
BLAS.

Regenerating fixtures and goldens needs the Python environment:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/golden/make_synthetic_openpmd.py   # synthetic openPMD dumps
.venv/bin/python tools/golden/make_scoring_fixtures.py    # CSV-only scoring campaign
.venv/bin/python tools/golden/generate_golden.py          # run the Python reference over both
ctest --preset release
```

The fixtures are analytic (no RNG in the field data, Halton sequences for
particles), so regenerating them byte-for-byte reproduces the same files.

## Checking against your own data

The synthetic fixtures cover the code paths, not the physics scale. To check a
real campaign, point the opt-in test at a copy that already has reference CSVs:

```bash
GUIDING_REAL_DATA_DIR=~/guiding_real_subset ctest --preset release
# or directly, with per-case detail:
GUIDING_REAL_DATA_DIR=~/guiding_real_subset build/release/cpp/tests/guiding_tests "[real]"
```

It accepts either a campaign directory or a single case, recomputes
`guiding_metrics.csv` from the HDF5 dumps and compares against the file that is
already there. Without the variable the test is skipped, so it never breaks a
normal run. A useful subset is one complete triplet with two or three dumps per
case, its `guiding_metrics.csv`, and `case.env` / `resolved_parameters.json`.

The same thing by hand, for one case:

```bash
guiding_cli case --diag CASE/diags/diag1 --outdir /tmp/check --overwrite
diff /tmp/check/guiding_metrics.csv CASE/guiding_metrics.csv && echo identical
```

## Performance

Measured on an i7-4800MQ laptop (8 threads, warm page cache), identical output:

| Workload | Python | guiding_cli |
|---|---|---|
| 12 field dumps, 3 × 1600 × 160 (212 MB) | 3.06 s | 0.28 s (1 thread), 0.16 s (8 threads) |
| 1 particle dump, 2 M macroparticles (107 MB) | 10.1 s (read + reduce) | 5.0 s (with CSVs) |

Where the speed comes from: each field component is read once (the reference
reads `E/r` and `E/t` twice each), only the positive-r half is built, dumps are
reduced in parallel, contiguous datasets bypass HDF5 with `pread`, and a
percentile sample is sorted once and reused for every percentile of that
dataset.

Knobs: `--threads`, `--no-raw-reads`, and `GUIDING_THREADS` in the environment.
Readiness scans use `readdir`/`statx` only, so a dry run never opens HDF5.

## Extending

The rule that keeps parity: whenever a value ends up in a CSV, mirror the
reference expression exactly, comment which Python line it comes from, and add
the case to a golden run. `docs/SPEC.md` records the behaviours that are easy
to get wrong (pairwise summation, NaN handling, `repr` text, dtype promotion).

- **A new metric per dump**: add the field to `physics::LaserMetrics` or
  `WakeMetrics`, compute it in `field_metrics.cpp`, append the column in
  `products/case_reduction.cpp` (`guiding_metrics_columns` and
  `format_guiding_metrics_csv`), and extend the Python side first so a golden
  exists to compare against.
- **A new CLI flag**: the matching `cpp/cli/*_command.cpp`; keep the Python
  name, and thread it through the `*Options` struct in `products/`.
- **A new panel**: add `draw_*_panel` in `cpp/gui/`, declare it in `panels.hpp`,
  call it from `App::draw_frame`, and dock it in `build_default_layout`. Panels
  only read snapshots from `DataStore`; anything that touches the file system
  belongs in the store, which keeps the UI thread free.
- **Debugging a mismatch**: `guiding_cli inspect --diag DIAG` prints the layout
  and dtypes; the parity tests print the first differing rows and columns with
  their relative difference.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HDF5 (C library) was not found` | `sudo apt install libhdf5-dev`, or `-DHDF5_ROOT=...`, or `-DGUIDING_FETCH_HDF5=ON` (the error message lists all three) |
| GUI fails with `glfwInit failed` | No display: run the CLI instead, or `xvfb-run -a guiding_gui ...` |
| GUI links fail on a compute node | Configure with `-DGUIDING_BUILD_GUI=OFF` (preset `hpc`) |
| `FetchContent` cannot download | Pre-populate with `FETCHCONTENT_SOURCE_DIR_<NAME>=...` or use a machine with network for the first configure |
| CMake older than 3.24 | Load a newer CMake module; presets need 3.24+ |
| A case shows "raw" but never "reduced" | The CSV is missing or lacks the triplet columns; the Log panel and `--min-h5` / `--min-last-h5-age-min` explain the readiness gate |
| Numbers differ from Python | Run the `[real]` test above; it prints the first differing cells. Differences beyond the two documented tolerances are bugs worth reporting |
