# Development branch review — 2026-10-20

Repository: `guiding_analysis_module`. Audited base: `972c1e6b84e43d3aafc50319c0a20f02ad5bad2c`.
Remote references observed at `2026-10-20T11:41:19.366190+00:00`.

## Evidence scope

Remote heads were fetched into a temporary bare shared clone. Source repositories
were not switched, merged, rebased, or updated with development code. Inspection
used `git show`, `git diff`, `git log`, `git rev-list --left-right --count`,
and `git merge-base --is-ancestor`. Reference acquisition used `git ls-remote
--heads origin` and `git fetch --no-tags origin
'+refs/heads/*:refs/remotes/origin/*'` in the temporary clone. No branch tests,
simulations, scheduler jobs, analysis pipelines, or cleanup commands were run.
The inventory is a dated remote snapshot; findings below cover selected relevant
implementations, not certification of every changed file.

`BASELINE_2026-10-20.md` is preserved byte-for-byte as historical evidence for
the audited base and its recorded Cloud environment. It does not validate these
development tips, future versions, SUNRISE, LYNX, or WarpX execution. Its SHA-256:
`a33a9b1ef4796c99dad2b6bfd979eadd05ac7bd9a5fb411b9d8d4ae6c2905fbb`.

## Remote branch inventory

Counts are commits unique to the audited base / unique to the remote branch.
A zero branch-only count denotes an ancestor or the identical base, not pending
implementation work. Hashes, rather than moving branch names, identify evidence.

| Remote branch | Commit | Base-only / branch-only |
| --- | --- | --- |
| `feat/clpu-soft50-v1` | `d8a42b79840935e829020ebb86dfc3bc4e1a1936` | 5 / 1 |
| `feature/beam-evolution-v1` | `da0ab3528eb41cbe80d05c91e4962f89e33c9117` | 0 / 2 |
| `feature/clpu-n2-beam-visualization` | `990ada876ac21a0ab77a0a94ec23f3fc98a27563` | 0 / 3 |
| `feature/clpu-n2-exact-exit` | `e809b43d2071e5fa5cb39de2613f3e9d170bea84` | 4 / 0 |
| `main` | `972c1e6b84e43d3aafc50319c0a20f02ad5bad2c` | 0 / 0 |
| `shared-baseline-dryrun` | `469940b18aaf2e94e1264e8ae4936c8d43669d78` | 38 / 0 |

## Verified implementation findings

- `feature/clpu-n2-exact-exit` is an ancestor of the audited main: main contains
  four additional commits and no branch-only commit remains. The resolved
  exact-exit behavior is already part of the baseline, not a missing feature
  to reimplement or merge again.
- `feat/clpu-soft50-v1` has diverged: five commits are unique to the audited
  base and one to this branch. Its three-dot diff adds resolved physical exit
  coordinates and an optional `--maximum-target-iteration-delta` alignment
  limit to `scripts/analyze_particle_case.py`, with tests. This older nearest
  frame/alignment path must not replace the current exact-resolved policy.
- `feature/beam-evolution-v1` adds beam time series and spatial integration;
  `feature/clpu-n2-beam-visualization` descends from it and adds combined N2
  populations and animations. `scripts/analyze_beam_evolution.py` preserves
  the union of species iterations, recording unavailable rows rather than
  treating missing diagnostics as empty populations. Combined statistics use
  concatenated particles, not averages of per-species statistics.
- Beam coordinates and exit sampling are separate contracts. The beam design
  uses absolute guiding coordinates, with explicit time interpolation and no
  extrapolation; exact particle exit sampling remains separately recorded.
  The beam documentation distinguishes full-plasma integrals from observed
  partial coverage. `--allow-partial-coverage` explicitly permits incomplete
  integration results; it does not suppress read/coordinate errors.
- The guiding branch documentation says the beam feature is not yet wired into
  workflow cleanup/optimization. That statement describes this repository's
  feature scope. The separately reviewed workflow tip
  `feature/clpu-n2-beam-sobol` already invokes beam analysis with partial
  coverage allowed and validates scheduled products. Its validator reports
  coverage without requiring complete spatial integration. Integration must
  reconcile both repositories at exact versions.

Evidence: `scripts/analyze_particle_case.py`,
`scripts/analyze_beam_evolution.py`, `docs/BEAM_EVOLUTION.md`, and
`docs/CLPU_N2_BEAM_VISUALIZATION.md` at the pinned tips. These are source and
contract inspections, not executed tests or validation of real diagnostics.

## Possible conflicts and decisions required

Both beam development branches change `README.md`, which this documentation
stage also changes. Reconcile the beam feature descriptions with the new
retention/exact-time guidance during any future merge. This is a confirmed
file overlap, not a claim that Git has reported an unresolvable text conflict.
`shared-baseline-dryrun` is an old ancestor, not outstanding development work.

The diverged Soft50 branch affects the same particle-selection responsibilities
as current exact exits. Preserve exact-resolution failures and distinguish
absolute position, propagation distance, time interpolation, and exact particle
iteration. Review whether hard-50 beam products are QA, new objectives, or
required retained products; they are not interchangeable with Soft50 acceptance.
Decide acceptable coverage and which future particle/field studies require raw
retention before any workflow cleanup policy changes.

## Integration boundary

The current stage is documentation only. No branch implementation, species
selection, objective, temporal policy, scientific-validation rule, or raw
cleanup behavior is adopted. Scientific decisions above remain pending.
Any later integration requires its own version-specific tests and site evidence.
