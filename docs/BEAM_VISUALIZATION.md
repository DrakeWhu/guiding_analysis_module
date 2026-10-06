# Complete beam visualizations

Entry point: `scripts/animate_beam_evolution.py`. SUNRISE launcher: `scripts/sunrise_beam_visualization.sbatch CASE_DIR OUTPUT_PARENT ABSOLUTE_SCRIPT_PATH`. The launcher creates `OUTPUT_PARENT/job-JOBID`. Create OUTPUT_PARENT before submitting so SLURM can write its log.

The script processes every particle/field frame, requires exact guiding coordinates and matching timestamps, and fails if frames disagree. It makes two full GIFs and all individual PNGs. GIF playback is 180 ms per saved frame, not physical time; titles show iteration and actual time.

Phase-space panels: forward electrons at all energies in (z-z_frame,K), forward electrons at all energies in (x,ux), and forward electrons with K>=50 MeV in (x,ux). Histograms show charge magnitude in pC per bin, weighted by physical macroparticle weights, projected over all azimuths. No particle sampling or energy lower cut is used. Histograms have finite bin resolution.

Density panels: x-z and y-z cuts, theta=0 and pi/2, reconstructed from all stored modes. They show n_e/n0=-rho_electrons/(e*n0). Negative reconstructed values are retained; truncated mode expansions may give signed artifacts. Density is a cut, while particle histograms are projections.

The first pass scans the entire series to determine fixed spatial, energy, momentum and colour scales. The second renders all frames. Spatial coordinates move with exact guiding z_max; no frame or displayed population is manually cropped. Extremely dense cold populations may dominate the all-energy panels; the separate energetic panel uses the same charge scale.

Outputs: phase_space_full.gif, rho_electrons_full.gif, phase_frames/*.png, rho_frames/*.png, frame_manifest.csv and visualization_metadata.json. The manifest permits checking Q50 against beam_evolution.csv. Existing output directories are refused.

Dependencies: numpy, matplotlib, openpmd_viewer, Pillow (PIL). FFmpeg is not required.

Local validation: synthetic two-frame rendering through the complete script, charge weights, reversed field-axis order, both PNG sets and both GIFs; bash syntax check. Real SUNRISE raw rendering remains to be validated.
