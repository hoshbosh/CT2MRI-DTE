#!/bin/bash

#SBATCH --job-name=ct2mri_train    # Job name 
#SBATCH --account=neurology-dept                    # Run on a single CPU 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=joshua.labasbas@ufl.edu     # Where to send mail 
#SBATCH --nodes=1                    # Run on a single CPU 
#SBATCH --gres=gpu:2                   # Run on a single CPU 
#SBATCH --cpus-per-task=12                    # Run on a single CPU 
#SBATCH --mem=350gb                     # Job memory request 
#SBATCH --time=32:00:00               # Time limit hrs:min:sec 
#SBATCH --output=ct2mri-train%j.log   # Standard output and error log 
pwd; hostname; date 

module purge 
module load python 
module load conda
module load cuda
module load gcc/14.2.0
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load freesurfer

conda activate ct2mri

./shell/train/train.sh

date
