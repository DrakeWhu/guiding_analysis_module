# General RZ visualization

The laser_guiding package reads existing WarpX openPMD/HDF5 diagnostics without
inferring a capillary, wall, gas exit, injection mechanism, vacuum baseline,
or self-guiding. The first example is ELBA L4 with a **preformed** helium channel.

This stage provides configurable field representations and electron histograms.
It does not implement validated laser/plasma separation, laser energy, a0,
confinement, or trapped-beam measurements.

## Compatibility and provenance

The base is main f6adaf9997e72f1a46de068d5fa0860830053b12.
The cap_guiding.openpmd_io compatibility API preserves public names, micrometre
units, the constructor patch point and CLPU field-call defaults. The general
reader accepts both RZ axis orders and explicit modes/angles. Existing CLPU
commands, scores, CSVs, campaign discovery and exact-exit policies are retained.

The general CLI requires neither guiding_metrics.csv nor warpx_used_inputs,
plateau bounds, triplets or CLPU filenames. Development branches da0ab35
(beam evolution) and 990ada8 (N2 visualization) informed this implementation;
their campaign-specific CLI contracts are not merged wholesale.

The SUNRISE inventory of 2026-11-25 placed main at
0c4b87b6c7c2476f71e50aeda8153f9160f21639, seven commits behind this base.
No HPC checkout is updated here. Validate in a separate checkout.

## Configuration

Specify diagnostic paths relative to --case-dir, or absolute paths. Unknown
configuration keys fail. The example examples/elba_l4_visualization.json reads
the exact laser_centroid_initial_z_m and laser_wavelength_m keys from the case's
resolved_parameters.json. Values of the form {"parameter": "nested.key"} resolve
explicit parameters; preflight-only parameter files (dry_run: true) fail.

| reference.kind | Horizontal coordinate |
| --- | --- |
| absolute | absolute z |
| linear | z - (initial_z_m + velocity_m_s * diagnostic_time_s) |
| window_center | z minus the midpoint of the saved grid |

A linear reference is nominal, not a measured laser peak. Views specify
theta_rad and transverse_um/longitudinal_um bounds in micrometres. The full
signed plane comes from openPMD-viewer, never a reflected positive half.
Views at the same angle reuse reconstructed fields. Spatial aspect is equal;
requested limits remain fixed, with blank regions where data were not saved.
A null axis bound uses the full extent across the selected frames, with fixed
limits for the entire sequence. The ELBA full view uses this for z.

| field kind | Quantity before unit_scale |
| --- | --- |
| field | named scalar or vector component in reader SI units |
| electron_density | -rho_species/e in m^-3; a rho_* scalar is required |
| transverse_e2 | Ex^2+Ey^2 in (V/m)^2 |

Each field has an explicit m (0, 1, ... or "all"), label, units and scale.
For transverse_e2, smooth_wavelengths requests a rolling spatial mean along z,
rounded to mesh cells with edge padding; wavelength_m is required. This proxy
contains plasma contributions and is neither isolated laser intensity nor a0.
No cylindrical volume or angular integration is performed. Vector conversion
and angular reconstruction are delegated to openPMD-viewer.

Linear, symmetric and logarithmic color limits are shared across selected
frames and views for each quantity; manual vmin/vmax are supported. Nonpositive
logarithmic samples are masked and their count is recorded, including repeated
samples in overlapping views. Use a signed scale to inspect those values.
Duplicate views at theta=pi/2 to compare planes; duplicate fields with different
m values and names to compare mode zero and all modes.

## Optional electron products

Fields are the default; the particle diagnostic is never opened unless
--with-particles is passed. This path is electron-only: every configured
species must be an electron population. Species names are explicit. Missing
species fail; registered empty species are recorded as empty per snapshot. The pure-He example requests only its background
electrons; add nitrogen/argon populations explicitly for injection cases.

openPMD-viewer returns u=gamma*beta. Kinetic energy uses
mc^2*u^2/(sqrt(1+u^2)+1) to retain cold-electron precision. Charge magnitude
is e*sum(w), with no extra r or 2*pi factor. Histograms sum physical charge
across species rather than averaging their statistics.

There is no direction or energy cut by default. The optional forward_only
and energy_min_mev selections are separate from histogram bin ranges.
Energy underflow/overflow and charge outside the 2D histogram are recorded.
The final bin includes its upper edge following NumPy conventions.

Products are dQ/dK in pC/MeV and a (relative z, K) map in pC/bin integrated
over transverse coordinates and angles. They describe electrons still in
the saved moving window, not a Cartesian slice, cumulative accelerated charge,
trapped charge or trajectories. No individual particle identifiers are used.

Particles are read once per frame/species and reduced to small NPZ histograms
reused for plotting. Fields use two streaming passes for common scales and
rendering, without retaining all field frames in memory.

## Exact snapshot selection

For a running simulation provide completed_log or --completed-log. Files must
be at least 600 seconds old by default, unchanged during metadata inspection,
and no later than a logged STEP ... ends. record. Normal HDF5 locking stays
enabled. Only selected files are exposed to the viewer. Source signatures are
checked again before declaring success. These checks preserve the earlier
ELBA live-reader policy, not a full raw-validation or completion declaration.

Without a log, --static-diagnostics explicitly declares closed input.
Age alone is not proof of completion. Requested iterations must exist exactly.
With --all-frames --with-particles, eligible iteration sets must agree;
differences fail rather than dropping frames. Times and field time offsets
must agree. The output directory must be new and outside raw diagnostics.
Failure may leave partial products, but no complete visualization.json.

## SUNRISE use

From a separate checkout in the proven Python 3.10 analysis environment:

~~~bash
source "$HOME/apps/env/campaign-workflow.sh"
source "$HOME/apps/venvs/guiding-analysis-py310/bin/activate"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CASE="$HOME/warpx_runs/elba_l4_he_2cm_20261124/000_He_L4_reference"

python -B scripts/visualize_case.py \
  --case-dir "$CASE" \
  --config examples/elba_l4_visualization.json \
  --outdir "$CASE/post/general_visualization_smoke_01" \
  --iterations 0,2631,10000
~~~

Those iterations come from the prior ELBA inspection: confirm availability in
the target run. Missing exact frames fail. First compare a few frames with
the independent ELBA reader. This example reads fields only. Use the existing
SLURM analysis wrapper for substantial analysis; the CLI does not submit jobs.
Keep analysis Python 3.10 separate from PIC Python 3.14.

After smoke validation, deliberately select a complete sequence and optional
electron products:

~~~bash
python -B scripts/visualize_case.py \
  --case-dir "$CASE" \
  --config examples/elba_l4_visualization.json \
  --outdir "$CASE/post/general_visualization_full_01" \
  --all-frames --with-particles --movie
~~~

MP4 encoding streams through imageio/ffmpeg; PNGs remain available. Frames have
uniform display cadence (default 6 fps), even when physical diagnostic spacing
differs. Image annotations and metadata retain physical times. Movies do not
resolve optical cycles between snapshots.

## Evidence and retention

Outputs include field PNGs per view, optional electron PNGs and NPZ histograms,
optional MP4s and visualization.json: resolved config, source signatures,
exact iterations/times, common field/spectrum scales, code hashes, versions and species charge
accounting. The CLI does not modify raw, simulation inputs, workflow state,
cleanup eligibility or existing reduced outputs.

New tests use real openPMD-viewer/h5py with synthetic files: both axis orders,
odd-mode parity, vector conversion, units, charge weights, empty/absent species,
exact selections, age/log guards, mutation, timestamps, fields-only behavior,
one particle pass, plots and actual MP4 encoding. Run the complete legacy suite
as well. CI and synthetic data are not validation of SUNRISE or ELBA physics.
A real-data comparison against the independent reader remains required.

References:

- https://warpx.readthedocs.io/en/26.05/dataanalysis/openpmd.html
- https://warpx.readthedocs.io/en/26.05/usage/parameters.html
- https://openpmd-viewer.readthedocs.io/en/latest/api_reference/generic_interface.html
