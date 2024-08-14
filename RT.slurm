#!/bin/bash -l
#SBATCH --account=pawsey1077
#SBATCH --job-name=RTGroupParameterFit
#SBATCH --nodes=2
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mail-user=23303755@student.uwa.edu.au
#SBATCH --mail-type=ALL
# load modules
module load python/3.10.10
module load py-numpy/1.23.4
module load py-matplotlib/3.6.2
module load py-pandas/1.5.1
# launch serial python script
srun python3 RTGroup.py
