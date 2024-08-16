#!/bin/bash

# Define parameters
SUBDIR="./"  # No spaces around '=' in variable assignments
N=5          # No spaces around '=' in variable assignments
script="slurm_run_ptyrad_optuna.sub"  # No spaces around '=' in variable assignments

# Parse the command-line argument for -n
while getopts n: flag
do
    case "${flag}" in
        n) N=${OPTARG};;
    esac
done

# Loop to submit the sbatch script N times
for i in $(seq 1 $N); do
    echo "Submitting $SUBDIR/$script"
    sbatch "$SUBDIR/$script"
    sleep 1
done

echo "Jobs are all submitted!"