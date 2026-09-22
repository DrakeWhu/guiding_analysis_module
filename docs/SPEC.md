# Numerical specification of the reduction pipeline

This document is the contract between the Python reference (`cap_guiding/`, `scripts/`) and the C++ port (`cpp/`).

The C++ code reproduces the reference outputs from the same inputs:
- **Numeric CSV cells:** within 1e-9 relative. Bit-identical wherever the reference arithmetic is deterministic.
- **Other cells:** integer, boolean and string cells, and the column order, are identical.

Golden outputs live in `cpp/tests/data/golden/`. Their `metadata.json` records the reference commit and library versions. Regenerate them with:

```bash
.venv/bin/python tools/golden/make_synthetic_openpmd.py
.venv/bin/python tools/golden/generate_golden.py
.venv/bin/python tools/golden/make_numpy_reference.py
```

The reference semantics are pinned to:
- openpmd-viewer 1.11, h5py backend
- numpy 2.4
- pandas 3.0

## 1. Campaign layout

| Item | Location / rule | Source |
|---|---|---|
| Case discovery | `cases_full.tsv` or `cases.tsv` in the campaign root, parsed as tab, then whitespace, then comma separated; BOM stripped; column aliases accepted. Otherwise sorted sub-directories. | `campaign.py:304-455` |
| Case type | tokens `chan`/`channel`, `uni`/`uniform`, `vac`/`vacuum`, split on `[_\-/\\]+` | `campaign.py:117` |
| Case keys | `_`-delimited tokens `f<n>`, `n<dens>`, `refn<dens>`, `L<x>mm`, `foc[mp]?<x>(um\|mm)`, `d<x>um`; `.` is replaced by `p`, then lower-cased | `campaign.py:15-29` |
| Field diagnostic | `CASE/diags/fields`, else `CASE/diags/diag1` | `diagnostics.py:44` |
| HDF5 count | recursive `*.h5` under the field diagnostic | `campaign.py:131` |
| Case metrics | `CASE_METRICS_ROOT/CASE_ID/guiding_metrics.csv` plus `guiding_singlecase_score.csv` | `workflows.py:46` |

**Readiness states:**
- *raw-ready*: `h5_count >= --min-h5`, and either `--min-last-h5-age-min <= 0` or the newest `*.h5` mtime is at least that old.
- *reduced-ready*: `guiding_metrics.csv` parses, has at least one row, and has every `REQUIRED_COLUMNS` entry.
- *usable*: raw-ready or reduced-ready.

**Triplets:**
- Key: `(f, L, foc, density)`; a vacuum case uses its `refn` value as the density.
- Matching: uniform on the full key; vacuum on the full key, else the first vacuum with the same `(f, L, foc)`.
- Order: channels first, then orphan uniforms, sorted by label `f…_n…_L…mm_d…um_foc…`.

## 2. openPMD read semantics (openpmd-viewer 1.11)

- **Iterations:**
  - Every `*.h5`/`*.hdf5` file directly inside the diagnostic directory is opened, and the integer keys of its `/data` group become iterations.
  - The list is sorted and then taken as `[::stride]`.
  - Every file must have `openPMD` starting with `1.`.
- **Paths:** `'/'.join(["/data/<it>", meshesPath, "E/r"])`, with `//` collapsed once.
- **Time:** `(iteration.time + record.timeOffset) * iteration.timeUnitSI`. For Cartesian `x`/`y`, the time and axes come from the **`t`** component.
- **Mesh shape:**
  - thetaMode datasets are `(Nm, Nz, Nr)` with `axisLabels == [z, r]`; any other order is rejected, as in `openpmd_io.rz_signed`.
  - Values are multiplied by `unitSI` when it is not 1. In-place multiplication keeps float32 arrays in float32.
- **Axes:**
  - For each axis: `step = gridSpacing[a]·gridUnitSI` and `start = gridGlobalOffset[a]·gridUnitSI + position[a]·step`.
  - The points are `np.linspace(start, start + (n−1)·step, n)`.
  - The signed radial axis is `concat(−r[::-1], r)`.
- **Mode sum at θ:**
  - `above = [1, cos(θ), sin(θ), cos(2θ), …]`; `below` multiplies mode `m` by `(−1)^m`.
  - Upper half: `Σ_k above[k]·F_k`. Lower half: `Σ_k below[k]·F_k`, radially reversed.
  - Each sum is computed in float64 and stored back in the dataset dtype.
- **Cartesian components:**
  - `Fx = cos θ·Fr − sin θ·Ft` and `Fy = sin θ·Fr + cos θ·Ft`, then the lower half is negated.
  - At θ = 0 this still multiplies by 0, so a NaN or inf in the other component propagates.
- **Positive r:**
  - `rz_positive` keeps the rows with `r·1e6 >= 0` in signed order.
  - When the radial grid starts on the axis (`position == 0`), this includes the mirrored `r = −0.0` row.
- **dtype promotion (NumPy 2):**
  - `np.cos(θ)·Fr` promotes float32 to float64, so `Ex`/`Ey` are float64.
  - A scalar or cylindrical component such as `E/z` keeps its dataset dtype.
  - NumPy 1.x kept `Ex` in float32. The goldens record which numpy produced them.
- **Particles:**
  - Positions are `position/c + positionOffset/c` in metres; momenta are `u = momentum/(mass·c)`.
  - The ED-PIC `macroWeighted`/`weightingPower` correction is applied.
  - Constant records (`value` plus `shape`) are expanded; output is float64.

## 3. Field reduction: `guiding_metrics.csv` (`metrics.py`)

Per iteration, on the positive-r grid `[row, z]`:

- **Intensity and weights:** `I = Ex² + Ey²`, `r_m = r_um·1e-6`, `z_m = z_um·1e-6`, `w = np.maximum(r_m, 0)`.
- **On-axis profile:**
  - `I_z[z] = Σ_rows I·w`. numpy accumulates this row by row, in row order, from `+0.0`.
  - `dz_um = median(diff(z_um))`.
  - `cells = max(1, round_half_even(smooth_um / dz_um))`.
  - `I_z_smooth = np.convolve(I_z, ones(cells)/cells, "same")`:
    - `out[m] = Σ_j y[m − ⌊cells/2⌋ + j]·(1/cells)`, over in-range indices only;
    - accumulated in index order;
    - skipped when `cells <= 1`.
- **Peak position:** `j_peak = argmax(I_z_smooth)` (first maximum; the first NaN wins), and `z_peak_um = z_um[j_peak]`.
- **Waist:**
  - Window: `|z_um − z_peak_um| <= 5`; if empty, only `j_peak`.
  - `P[row] = Σ_window I`, accumulated in z order from `+0.0`.
  - `denom = pairwise_sum(P·w)`.
  - `waist_um = sqrt(2·pairwise_sum((r_m²·P)·w)/denom)·1e6` when `denom > 0`, otherwise NaN.
- **Intensity metrics:**
  - `energy_proxy = pairwise_sum(flatten_C(I·w))·median(diff(r_m))·median(diff(z_m))`.
  - `peak_I_proxy = max(I)`, which is NaN if any value is NaN.
  - `Eperp_peak_Vm = sqrt(peak_I_proxy)`.
- **a0:**
  - `a0_peak = e·Eperp/((m_e·c)·ω0)` with `ω0 = (2π·c)/λ0`.
  - Constants: e = 1.602176634e-19, m_e = 9.1093837015e-31, c = 299792458, λ0 = 0.8e-6. NaN when `Eperp <= 0` or the input is non-finite.
- **Margins:** `front_margin_um = max(z_um) − z_peak_um`, `back_margin_um = z_peak_um − min(z_um)`.
- **Wake:**
  - On the `E/z` row at `argmin(|r_um|)`, over `z_um ∈ [z_peak − wake_behind_um, z_peak − wake_gap_um]`: max, min, `max|Ez|` with its z (first maximum), `z − z_peak`, and `rms = sqrt(mean(Ez²))`.
  - For float32 `E/z`, the square, mean and sqrt are float32 operations.
  - All six values are NaN when the window is empty.
- **Row fields:**
  - `iteration`.
  - `time_fs = time_s·1e15` (NaN if non-finite).
  - `z_min_um`/`z_max_um` from the `Ex` axis.
  - `propagation_mm = (z_max_um − z_max_um[0])·1e-3`.
  - `z_peak_relative_um = z_peak_um − z_min_um`.
- **Failure:** the case fails with *"No valid laser dumps found"* unless some row has finite z_peak/waist/peak_I/energy with `peak_I > 0` and `energy > 0`.

`pairwise_sum` is numpy's reduction for 1-D contiguous float arrays:
- `n < 8`: a sequential sum from 0.
- `n <= 128`: eight accumulators, combined as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, followed by the tail.
- Above 128: split at `n/2` rounded down to a multiple of 8.

`np.median` sorts, then averages the middle slice as `(0 + a)/1` or `((0 + a) + b)/2`. It returns NaN if any value is NaN.

Columns, in order:

```
iteration,time_fs,z_min_um,z_max_um,z_peak_um,front_margin_um,back_margin_um,waist_um,peak_I_proxy,
Eperp_peak_Vm,a0_peak,energy_proxy,Ez_wake_max,Ez_wake_min,Ez_wake_absmax,Ez_wake_rms,z_Ez_absmax_um,
z_Ez_absmax_rel_um,propagation_mm,z_peak_relative_um
```

## 4. Single-case score v1: `guiding_singlecase_score.csv` (`singlecase_guiding.py`)

- **Plateau window,** first rule that applies:
  1. explicit start and end;
  2. explicit length, starting at 5 mm;
  3. the first `L<x>mm` token in the CSV path, giving `(5, 5 + x)`;
  4. otherwise the case fails.
- **Rows:**
  - Sort by `propagation_mm` with NaN last.
  - Plateau rows have propagation inside the window.
  - Valid rows have finite positive `a0_peak` and `waist_um`; at least 3 are required.
- **Coverage:** `coverage = clip01(span/length)·(n_valid/n_plateau)`.
- **Entry and exit windows:**
  - `[start, start+1]` and `[end−1, end]` over the valid rows.
  - When empty, fall back to the first or last valid row.
  - Each window value is the median of its finite values.
- **Components:**
  - a0 retention: `clip01(a0_exit/a0_entry)`.
  - a0 stability: `exp(−(rms(max(0, log(a0[k]/a0[k+1])))/log 1.3)²)`.
  - Waist growth: `exp(−(max(0, log((waist_max/waist_entry)/1.1))/log 1.5)²)`.
  - Waist stability: `exp(−(std(log waist)/log 1.35)²)`, with `std` using ddof = 0 and pairwise sums.
- **Score:** `100·coverage·Σ wᵢ·cᵢ`, with weights `[0.35, 0.15, 0.45, 0.05]` normalised by their pairwise sum.
- **Output:** a single row written by `pandas.DataFrame([result]).to_csv(index=False)`. Keys follow Python dict order, and each failure path emits its own subset of keys.

## 5. Particle products (`particles.py`, `transverse.py`, `beamlike.py`, `soft50.py`, `particle_exit.py`)

**Reading** (openpmd-viewer `read_species_data`, h5py backend):
- The component is read as float64 (`output_type`). `weighting`, `positionOffset` and `mass` keep their stored dtype.
- **ED-PIC correction:** when `macroWeighted == 1` and `weightingPower != 0`, apply `data *= w ** (-weightingPower)`.
  - numpy's `fast_scalar_power` handles −1, 0.5 and 2 as reciprocal, sqrt and square.
  - The exponent is an HDF5 attribute, so it is a numpy scalar and takes part in NumPy 2 promotion. float32 weights with a float64 `weightingPower` give float64 `1/w`. Only a float32 or small-integer attribute keeps float32.
  - `data *= unitSI` behaves the same way.
- **Momentum:** `u = p · (1/(m·c))`, applied only when every mass is non-zero. For a float32 mass dataset, `m·c` stays float32 because `c` is a Python float.
- **Time:** `time_fs = time·timeUnitSI·1e15` (no `timeOffset`). `avail_species` and record components come from the first file only.

**Scopes and rows** (`scripts/analyze_particle_case.py`):
- **Scopes:** with several `--species`, the first scope is `all_electrons` (the species concatenated), followed by one scope per species in the order given. With one species, its name is the only scope.
- **Summary row:** `{**selection_info, **summarize_dump(...)}`. Columns come from the first row's keys, in dict order. Beamlike columns are merged first; transverse and then soft50 columns overwrite them in place, so the seven empty transverse placeholders keep their positions.
- **Weighted statistics:**
  - `np.average` = `pairwise(a·w)/pairwise(w)`.
  - The percentile does no interpolation: sort by value, take the cumulative sum, then `searchsorted(left)` at `p/100·total`.
- **Scalar and array powers:**
  - A float scalar `** 2` (Python float or `np.float64`) calls libm `pow`, which is not always equal to `x*x` (about 6 in 10⁴ random operands differ). C++ uses `np::c_pow`, with the exponent kept out of reach of constant folding.
  - An array `** 2` is `np.square`, i.e. `x*x`.
- **Acceptance:** rows for θ cuts × E cuts over the sorted unique grids; `accepted_charge_pC = Σw·e/1e-12`.
- **soft50:** one curve row per sorted unique `energy_low`; missing columns are written as empty strings.

**Iteration selection:**

| `--which` | selection_info keys (in order) |
|---|---|
| `last` | `selection_mode`, `selected_particle_iteration` |
| `all` | `selection_mode`, `analysis_stride` |
| `exit` (legacy) | `selection_mode`, `exit_kind`, `guiding_metrics_csv`, `target_propagation_mm`, `target_guiding_iteration`, `target_guiding_propagation_mm`, `selected_particle_iteration`, `target_iteration_delta`, `available_particle_iterations_{min,max}`, `n_available_particle_iterations` [, `maximum_target_iteration_delta`, `target_iteration_alignment_status`] |
| `exit` + `--resolved-parameters` | `selection_mode` (= `exit`, overwritten in place), `exit_kind`, the target keys, the exact-iteration keys, `guiding_context_available`, `guiding_metrics_csv` [, the guiding keys, `particle_vs_guiding_iteration_delta`] |

- **Legacy target**, first match wins:
  1. `--target-propagation-mm`.
  2. `CASE/resolved_parameters.json`: `(plateau_end_z | plasma_end_z) − plasma_start_z`.
  3. `case.env`:
     - Lines are `KEY=VALUE`; an `export ` prefix is allowed.
     - The value is the first `shlex.split` token. If shlex fails, the value is the text with quotes stripped.
     - Plateau length comes from `PLATEAU_LENGTH_MM`…, else `CAP_PLATEAU_LENGTH_M`… ×1e3.
  4. `_L<x>mm_` in the case name.
- **Capillary target:** plateau + `--downramp-mm`, else `CAPILLARY_LENGTH_MM`…, else plateau + `DOWNRAMP_LENGTH_MM`…, else plateau with a `[WARN]`.
- **Nearest iteration:**
  - The guiding row is the first minimum of `|propagation_mm − target|`, with values parsed like pandas.
  - The particle dump is the first minimum of `|it − target_it|`, so ties go to the earlier dump.
- **Case directory:** `DIAG/../..` when DIAG's parent is named `diags`, else `OUTDIR/..`. Both are lexical, as in pathlib.

## 7. Scoring layers (`scoring.py`, `beamlike_pairs.py`, `joint_scores.py`)

These consume reduced CSVs only, so they are pure table-to-table transforms.

**Guiding score v2** (`score_case_csv`, over `guiding_metrics.csv`):
- Windows around the plateau `(start, end)` inferred from the CSV path: entry `[start, start+1]`, exit `[end−1, end+2]`, analysis `[start, end+2]` mm (configurable).
- `a0_exit`, `waist_entry_um`, `waist_exit_um` are medians of the finite values in their window; `a0_max_analysis` is the first `nanargmax`; `waist_jitter_log` is `std(log waist)` over finite positive values (ddof = 0, at least 2 of them).
- `valid_fraction` is the share of analysis rows with finite a0 and waist.
- Components: `clip(a0_exit/target, 0, cap)`, `clip(a0_exit/a0_max, 0, 1)`, `exp(−(max(0, growth−1)/σ)²)`, `exp(−(jitter/σ)²)`; `score = 100·Σwᵢcᵢ·valid_fraction` with weights .50/.20/.20/.10.
- Every failure is a row with `status = "failed"` and a reason (`missing_csv`, `empty_csv`, `missing_columns: [...]`, `could_not_infer_plateau_window`, `empty_*_window`, `non_finite_score_metric`, `non_positive_reference_metric`), never an exception.

**Triplet score** (`score_triplet_csvs`): the channel score times a reference factor.
- The reference is the case with the larger `a0_exit` (uniform wins ties).
- Row-wise mode: over the iterations common to all three cases inside the exit window, the medians of `log(a0_channel/a0_reference)` and `log(waist_reference/waist_channel)`. Without common iterations it falls back to the per-case exit medians, recording `reference_factor_mode = "exit_median_fallback"` and the reason.
- `combined = 0.8·a0 + 0.2·waist`, clamped down to the a0 advantage when the channel loses a0 beyond the deadband, then `tanh(deadband(combined)/scale)` with `deadband = log 1.05` and `scale = log 1.5`.

**Beamlike pairs** (channel vs uniform `particle_summary.csv`):
- Row selection `single` (strict), `last` (largest `iteration`) or `max-beamlike`; ties take the last row.
- Summaries without `beamlike_score` get the beamlike columns recomputed (`beamlike_score_source = "computed_on_the_fly"`).
- `log((score_channel + floor)/(score_uniform + floor))` → `tanh` of the deadbanded advantage → `gain = max(channel, uniform)·factor`; the transverse-quality comparison repeats this on `beam_transverse_quality_score` and keeps its own bucket.
- Buckets: positive / neutral / negative by the sign of the factor, failed otherwise. Files are `csv.DictWriter` with LF rows, the preferred column order first.

**Joint** (`joint_scores.py`): the triplet scores joined with the pair scores on `(channel_case_id, uniform_case_id)`, columns prefixed `guiding_`/`beam_`.
- `joint_bucket` and `triple_bucket` combine the guiding, beam and transverse buckets; `joint_positive_score = sqrt(guiding_final·beam_gain)` and `triple_positive_score = (guiding_final·beam_gain·transverse_gain)^(1/3)` over values floored at 0.
- Correlations use `np.corrcoef` (Pearson) and `scipy.stats.spearmanr` (average ranks) on the rows where both values are finite, needing at least 3.
- `pandas.merge` ordering for unique keys: inner and left keep the left order, right keeps the right order, outer sorts the keys. `value_counts` sorts by count, keeping first-appearance order for ties.

## 8. CSV text

| Writer | Line end | Float | NaN | bool | None |
|---|---|---|---|---|---|
| `csv.DictWriter` (guiding_metrics, particle CSVs, campaign reports) | `\r\n` | `repr` | `nan` | `True`/`False` | empty |
| `DataFrame.to_csv` (single-case score, triplet tables, score tables) | `\n` | `repr` | empty | `True`/`False` | empty |

- **Python `repr`:** shortest round-trip digits. It uses exponent notation when `decpt <= −4` or `decpt > 16`, e.g. `1e-05`, `1e+16`, `1000000000000000.0`.
- **Quoting:** `QUOTE_MINIMAL`, which quotes fields containing `,`, `"`, `\r` or `\n`, plus a lone empty field.

## 9. Compatibility quirks (kept on purpose)

- **Plateau-token separators:** `case_metadata._PLATEAU_TOKEN_RE` accepts `\`, `/`, `s`, `S`, `_` and `-` around `L<x>mm`. It does not accept whitespace, because `\\s` in a raw string is a literal `s`. A `P` fraction separator (`L2P5mm`) raises `ValueError`.
- **HDF5 counting:** readiness counts `*.h5` recursively, while the series listing only reads top-level `*.h5`/`*.hdf5` files.
- **Duplicate iterations:** if two files carry the same iteration, the later one wins. C++ sorts filenames to make "later" deterministic; Python uses `os.listdir` order.
- **Multi-species campaign directory:** `analyze_particle_campaign.py` resolves the particle diagnostic directory from the raw `--species` text. With `a,b`, it looks for `diags/a,b`, unless `--particle-diag-name` is given.
- **numpy's vectorized tanh:** `np.tanh` uses numpy's own AVX2 kernel, which differs from libm by up to 1 ULP, so `reference_factor` (and `final_score`) in the triplet scores are compared with a 1e-12 tolerance instead of bit for bit. `math.tanh` in `beamlike_pairs.py` is libm and stays exact.
- **Correlations:** `np.corrcoef` sums through BLAS, whose blocking the C++ port does not reproduce; `pearson` and `spearman` are compared with a 1e-12 tolerance.
- **Missing meshes group:** `describe_series` calls `list(None)` when a particle series has no meshes group, so Python fails. C++ prints `'avail_fields': []` and continues.
