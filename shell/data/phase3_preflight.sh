#!/bin/bash
#SBATCH --job-name=p3_preflight
#SBATCH --account=neurology-dept
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --output=p3-preflight-%j.log
#
# Pre-launch checks for the Phase 3 arms. CPU only, no GPU requested.
#   sbatch shell/data/phase3_preflight.sh
#
# Override the defaults by exporting first, e.g.
#   sbatch --export=ALL,HDF5=/path/to/other.hdf5 shell/data/phase3_preflight.sh
set -euo pipefail
pwd; hostname; date

HDF5="${HDF5:-/blue/neurology-dept/jlabasbas/hdf5s/fine_v2/256_train_axial.hdf5}"
N_SLICES="${N_SLICES:-2000}"
N_BLOCKS="${N_BLOCKS:-25}"

module purge
module load python
module load conda
conda activate ct2mri

cd "$SLURM_SUBMIT_DIR"

echo "=== preflight on ${HDF5}"
python -u runners/phase3_preflight.py \
    --hdf5 "$HDF5" \
    --n_slices "$N_SLICES" \
    --n_blocks "$N_BLOCKS"

date
