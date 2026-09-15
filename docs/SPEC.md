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

## 5. CSV text

| Writer | Line end | Float | NaN | bool | None |
|---|---|---|---|---|---|
| `csv.DictWriter` (guiding_metrics, particle CSVs, campaign reports) | `\r\n` | `repr` | `nan` | `True`/`False` | empty |
| `DataFrame.to_csv` (single-case score, triplet tables, score tables) | `\n` | `repr` | empty | `True`/`False` | empty |

- **Python `repr`:** shortest round-trip digits. It uses exponent notation when `decpt <= −4` or `decpt > 16`, e.g. `1e-05`, `1e+16`, `1000000000000000.0`.
- **Quoting:** `QUOTE_MINIMAL`, which quotes fields containing `,`, `"`, `\r` or `\n`, plus a lone empty field.

## 6. Compatibility quirks (kept on purpose)

- **Plateau-token separators:** `case_metadata._PLATEAU_TOKEN_RE` accepts `\`, `/`, `s`, `S`, `_` and `-` around `L<x>mm`. It does not accept whitespace, because `\\s` in a raw string is a literal `s`. A `P` fraction separator (`L2P5mm`) raises `ValueError`.
- **HDF5 counting:** readiness counts `*.h5` recursively, while the series listing only reads top-level `*.h5`/`*.hdf5` files.
- **Duplicate iterations:** if two files carry the same iteration, the later one wins. C++ sorts filenames to make "later" deterministic; Python uses `os.listdir` order.
