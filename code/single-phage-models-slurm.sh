#!/bin/bash
#
# Single phage models training Slurm script
# 
# This script calls several times my python script that runs a single iteration
# of the Monte Carlo Cross Validation of the single phage models.
#
#SBATCH --job-name=single-phage-models
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=100
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
python -u single-phage-models.py "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK"

echo "Task $SLURM_ARRAY_TASK_ID finished"
