# Repository guidance: guiding_analysis_module

## Scope and products

- `cap_guiding/` and `scripts/` analyze existing RZ WarpX/openPMD diagnostics.
  They do not launch WarpX or own workflow state and raw-data cleanup.
- Keep raw diagnostics, durable case-local metrics, and derived campaign
  comparisons distinct. A valid `guiding_metrics.csv` alone does not establish
  that every particle or future-analysis product has been preserved.
- A triplet links separate channel, uniform, and vacuum cases. Do not copy
  baseline measurements into a channel case's own metrics.
- Read the README's retention and temporal-selection notes before changing
  outputs. Coordinate schema changes with campaign-workflow and
  campaign-optimizer without introducing direct runtime coupling.
- Consult `docs/DEVELOPMENT_BRANCH_REVIEW_2026-10-20.md` before overlapping
  work: exact exits are already in the baseline; beam evolution and older
  diverged selection paths need separate integration review.

## Scientific invariants

- Preserve units, axes, species, weighting, cut definitions, and metadata.
  Preserve the resolved exact-exit policy and its failure when the required
  particle iteration is absent; do not silently choose a nearby frame.
- Keep particle selection and the corresponding guiding iteration explicit.
  A final-frame measurement is not the maximum achieved during propagation.
- Preserve particle summaries, acceptance curves, and species-specific
  products needed after raw removal. Their required/optional status is a
  campaign-level scientific decision.
- Analysis must not delete raw diagnostics. Use the external workflow's
  validated retention/cleanup contract for authorized cleanup.

## Verification

- Use `PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v`
  with the scientific environment, preferably in an isolated export.
- See `docs/BASELINE_2026-10-20.md` for dependencies and the dated test result.
  It is historical Cloud evidence, not validation of later revisions or HPC.
  Report unavailable openPMD support and skipped coverage explicitly.
- Separate unit/synthetic evidence from validation with real diagnostics.
  Do not run simulations or clean campaign data without task authorization.
- Write code, comments, tests, documentation, commits, and PR text in English.
