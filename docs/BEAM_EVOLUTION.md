# Physical beam evolution contract (v1)

This Python extension starts from the reconciled `main` commit
`972c1e6b84e43d3aafc50319c0a20f02ad5bad2c` (PR #1). It does not depend on,
repair or modify the C++ PR #2. Existing particle/guiding outputs and scores
keep their definitions. New products have schema `beam_evolution_v1`.

## Products and population

`CASE/beam_analysis/beam_evolution.csv` has one row per particle iteration and
requested electron species. `beam_evolution_summary.csv` has one row per species.
The species remain separate, including electrons produced by ionization.
Different diagnostics are read independently; the union of their iterations is
retained. A missing species frame is **unavailable**, not an empty population.

Selection is kinetic energy **>= 50 MeV** (configurable), `u_z > 0`, finite
position/momentum/weight and positive weight. There is no cone, radius or
particle-relative longitudinal window. Metrics use physical macroparticle
weights, not soft energy weights. They describe the stored diagnostic population;
filters, subsampling or resampling in the simulation cannot be undone here.
Invalid active records are reported; their frame is excluded from integration.
The implementation expects electron species and lab-frame particle diagnostics.

## Coordinates and plasma interval

`z_frame_um` is **absolute `z_max_um` from guiding**, the last longitudinal mesh
sample used by the existing propagation axis. It is not the physical box boundary,
laser peak, bunch centroid, distance since the first dump, or a particle crossing.
`z_centroid_um` is separately the charge-weighted particle centroid.

Equal iterations use an exact association, with timestamp consistency checked
when available. Particle frames between guiding dumps use bracketed linear
interpolation in physical time, marked `guiding_time_interpolated`; optional
`--max-coordinate-gap-um` caps that bracket distance. No coordinate is extrapolated
or guessed as `c*t`. The interpolation assumes linear moving-window advance
between the reference dumps; it does not improve the particle dump cadence.

The integration interval is the full plasma, including both ramps. Boundaries
are read from **`plasma_start_z`, `plasma_end_z` (metres)** in
`resolved_parameters.json`, converted to um. Without those keys, both absolute
`--plasma-start-um` and `--plasma-end-um` are required. No ramp length, plasma
origin or plateau length is guessed from a case name. Explicit bounds must agree
with resolved bounds when both are supplied. This accommodates scanning ramps.

The frame axis is a window reference for simultaneous snapshots. Even at the
plasma end it is not an integrated output-plane measurement. The recorded exact
`particle_diagnostic_targets.capillary_exit.iteration`, when available, selects
the exit snapshot independently of the frame axis; no nearest/last fallback is
used. An explicit `--exit-iteration` is supported when the target is absent.
Without either target, exit observables remain NaN (`not_requested`). No reduced
snapshot charge is labelled as charge integrated through an extraction plane.

## Moments and units

All averages divide by sum(w), without a sample Bessel correction. Particle
positions enter in m and normalized momenta are `u = p/(m_e*c) = gamma*v/c`.

- Charge is positive electron charge magnitude in pC, `e * sum(w) * 1e12`.
- Effective count is `(sum(w))**2 / sum(w**2)`.
- Normalized projected RMS emittance per plane is
  `sqrt(var(x)*var(ux) - cov(x,ux)**2)`, displayed in mm mrad (multiply m by 1e6).
  Existing `summarize_transverse_metrics` supplies it; old columns are unchanged.
- Plane angles are `atan2(ux,uz)`, `atan2(uy,uz)`. Their weighted means describe
  pointing; weighted standard deviations describe spread about the angular
  centroid. Both are in mrad. Uncentered angle RMS is also stored explicitly.
  These are plane statistics, not an exact coordinate-invariant cone angle.
- Position means, centered RMS sizes and lab-axis radial RMS are in um.
  Position/momentum covariances in um help identify correlated expansion.
- Kinetic-energy mean, RMS spread and p10/p50/p90/p95 are in MeV. Relative spread
  is RMS spread / mean kinetic energy, not RMS gamma / mean gamma.
  Spectral statistics are conditional on the selected >= 50 MeV population.
  Quantiles reuse the existing inverse weighted CDF: first value at cumulative
  weight >= requested fraction, without interpolation between weight centres.

An empty population has zero charge and undefined (NaN) quality. Read failures
have NaN charge. Moments from undersampled positive populations are retained and
flagged `insufficient_statistics`, excluded from quality integrals. The default
minimum is 3 particles and effective count >= 3; this prevents the most elementary
rank degeneracy, **not** a claim of adequate resolution or statistical convergence.
It is configurable upward and recorded in both products. Mathematical zero
emittance in a degenerate sampled distribution is retained as such.

Reference definitions: [WarpX 26.05 `BeamRelevant` and particle diagnostics](https://warpx.readthedocs.io/en/26.05/usage/parameters.html).
This diagnostic averages its species; our explicit energetic subset is separate.

## Temporal summaries and missing coverage

Integrate over absolute z in um:

`I_Q = integral Q(z) dz` (pC um),

`A_mean_Q = integral Q(z)*A(z) dz / integral Q(z) dz`.

Stored quality means: normalized emittances, centered angular dispersions,
absolute and relative energy spread. Each is an independent physical observable;
no combined score, acceptance or BO objective is constructed.

Integration uses trapezoids of **Q and Q*A separately**, with linear interpolation
of those quantities to bracketed plasma bounds. It never extrapolates at the start
or end, never bridges an unavailable/invalid row and never pools particle energies
from different frames. Empty frames contribute zero Q*A, despite undefined A.
Outside-plasma frames may bracket a boundary but contribute no outside length.

`--max-gap-um` rejects overly separated frames. The default is 1.5 times their
median positive spacing (recorded), a cadence heuristic. It cannot detect uniform
subsampling or every missing frame; use a limit justified by the actual diagnostic
cadence. Missing frames shared by all species are detectable only through gaps or
an independently known expected schedule. The module does not infer that schedule.

Canonical `charge_distance_pC_um` is NaN unless the full plasma is covered.
`charge_distance_observed_pC_um` retains the integral on valid observed segments.
Every quality metric has its own coverage fraction and charge-distance support;
its canonical mean is NaN for incomplete support. The `_observed` mean refers
only to supported segments and must not be treated as a full-plasma result.
Coverage is spatial support under the declared interpolation/cadence policy,
not proof that every physical feature has been resolved. A very sparse but fully
bracketed series can still have full coverage.

An existing, valid zero-charge exit is different from an unavailable exact exit.
Low-statistics exit moments carry `exit_quality_status`. Growth of divergence
followed by charge loss motivates a transverse-loss hypothesis, but this module
does not establish losses through a boundary or dephasing. In a force-free linear
drift with common longitudinal momentum, angles and projected normalized
emittance remain unchanged while beam size can grow.

## Running on existing diagnostics

Use the established SUNRISE `guiding-analysis-py310` environment documented in
the README, with Python loaded before activating the venv. No simulation jobs,
cleanup policy or optimizer configuration are changed by this command.

```bash
python scripts/analyze_beam_evolution.py \
  --case-dir /path/to/CASE \
  --species electrons,ionized_electrons
```

Known diagnostic directory resolution is reused. For nonstandard locations:

```bash
python scripts/analyze_beam_evolution.py \
  --case-dir /path/to/CASE \
  --species electrons,ionized_electrons \
  --species-diag electrons=/path/to/background_diag \
  --species-diag ionized_electrons=/path/to/ionized_diag
```

Campaign mode uses each case's own guiding CSV, resolved plasma bounds and exact
exit metadata (important when ramps are scanned):

```bash
python scripts/analyze_beam_evolution.py \
  --campaign-root /path/to/iteration \
  --species electrons,ionized_electrons
```

Defaults: all available particle frames, no plots, output in `beam_analysis`.
Existing new outputs require `--overwrite`; historical particle outputs are
untouched. Partial products are written for inspection with a nonzero exit code.
Insufficient-statistics quality flags do not discard valid measured charge or
make the command fail. The products are not yet wired into workflow validation,
cleanup or BO: those consumers must enforce the new coverage/status contract.

## Validation and C++ port

```bash
python -m unittest discover -s tests
```

Tests include weighted moments, hard/forward selection, pointing vs divergence,
linear drift invariance, spectral evolution without frame pooling, clipped ramp
intervals, zero/unknown/undersampled populations, gap/truncation coverage and exact
exit selection. An end-to-end suite creates actual synthetic openPMD/HDF5 files
for two species, including registered-empty and missing frames, and preserves the
legacy products. Real-data validation on full SUNRISE dumps remains necessary;
the fixed >=10 MeV tracking cohort is not the full per-frame energetic population.

Port the explicit array formulas in `summarize_beam_frame`, then the ordered-row
trapezoids in `summarize_beam_evolution`. I/O, CLI and optimisation policy are
separate. No IDs are converted or used by these population metrics; tracking
requires an independent integer-ID product. CSV schema/units/status names,
selection parameters and weighted-quantile convention form the porting contract.
