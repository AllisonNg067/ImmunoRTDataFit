#!/bin/bash -l
#SBATCH --job-name=controlGroupParameterFit
#SBATCH --nodes=2
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# load modules
module load python/3.10.10
module load py-numpy/1.23.4
module load py-matplotlib/3.6.2
module load py-pandas/1.5.1
# launch serial python script
srun python3 newControlGroup.py
