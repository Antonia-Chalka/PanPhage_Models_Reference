#!/bin/bash
#
# Recursive feature elimination with cross-validation Slurm script
# 
# This script calls several times my python script that runs a single RFECV. It
# starts from the features selected by a RFE
#
#SBATCH --job-name=rfecv_stats
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-128
#
########################################################################

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

date
echo "Hello from Slurm job array task: $SLURM_ARRAY_TASK_ID"
echo "Parallelising the CV across $SLURM_CPUS_PER_TASK CPUs"

./rfecv.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK

echo "Task $SLURM_ARRAY_TASK_ID finished"
