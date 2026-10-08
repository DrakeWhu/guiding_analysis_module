# N2 beam diagnostics

`analyze_beam_evolution.py --species preionized_background_electrons,nitrogen_ionized_electrons --combined-scope` writes per-species metrics plus the concatenated `all_electrons` population. Combined emittance and angular spread are recalculated from particles, not averaged between species. The selection is kinetic energy >=50 MeV and uz>0.

`--allow-partial-coverage` permits explicitly labelled partial spatial integration when all particle reads and coordinates succeed. It does not suppress missing species, invalid particles, missing exact exits, or coordinate errors. The default remains strict. Coverage is recorded: the guiding window upper bound need not bracket the entire absolute plasma interval at the earliest frame.

`animate_beam_evolution.py --species preionized_background_electrons,nitrogen_ionized_electrons --rho-fields rho_preionized_background_electrons,rho_nitrogen_ionized_electrons --discard-pngs` concatenates particle arrays, sums the two electron rho fields after checking their grids, and preserves every synchronized saved frame in both GIFs. PNGs are discarded after encoding; metadata and the charge manifest remain. Default single-species usage is preserved.
