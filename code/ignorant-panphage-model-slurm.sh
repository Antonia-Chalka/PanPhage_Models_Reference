#!/bin/bash
#
# Ignorant-PanPhage model training Slurm script
# 
# This script calls several times my python script that runs a single iteration
# of the Monte Carlo Cross Validation of the ignorant-PanPhage model.
#
#SBATCH --job-name=ignorant-panphage-model
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1
#
########################################################################

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

date
echo "Hello from Slurm job array task: $SLURM_ARRAY_TASK_ID"
echo "Parallelising the Monte Carlo CV across $SLURM_CPUS_PER_TASK CPUs"

source ../.venv/bin/activate
python -u ignorant-panphage-model.py "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK"

echo "Task $SLURM_ARRAY_TASK_ID finished"
