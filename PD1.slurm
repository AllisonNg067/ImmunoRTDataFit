#!/bin/bash -l
#SBATCH --job-name=RTGroupParameterFit
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:30:00
# load modules
module load python/3.10.10
module load py-numpy/1.23.4
module load py-matplotlib/3.6.2
module load py-pandas/1.5.1
# launch serial python script
srun python3 PD1Group.py
