#!/bin/bash
#
# Recursive feature elimination Slurm script
# 
# This script calls several times my python script that runs a single RFE
#
#SBATCH --job-name=rfe_stats
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=4G
#SBATCH --array=1-128
#
########################################################################

date
echo "Hello from Slurm job array task: $SLURM_ARRAY_TASK_ID"

./rfe.py $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID finished"
