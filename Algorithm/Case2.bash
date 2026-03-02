#!/bin/bash
#SBATCH --job-name=MATRIX-PROBLEM-Case2
#SBATCH --partition=cpu-medium
#SBATCH --mail-user=andrea.dinezza@polito.it
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.txt
#SBATCH --time=6-00:01:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16

module purge
module use /opt/share/sw2/modules/all/
module load Python
module load SciPy-bundle matplotlib

python Case2.py
