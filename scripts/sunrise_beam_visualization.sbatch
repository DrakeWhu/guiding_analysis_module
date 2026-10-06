#!/bin/bash
#SBATCH --job-name=beam002_movies
#SBATCH --partition=T1H
#SBATCH --account=eli-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=01:00:00
set -euo pipefail
module purge
module load GCC/12.1.0
module load Python/3.10.12
module load OpenBLAS/0.3.31
module load warpx/26.05-gcc12-openmpi413-all-dims
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source "$HOME/apps/venvs/guiding-analysis-py310/bin/activate"
python -u "$3" --case-dir "$1" --outdir "$2/job-$SLURM_JOB_ID"
