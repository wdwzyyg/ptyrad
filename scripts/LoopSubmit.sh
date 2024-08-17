#!/bin/bash

# Define parameters
# No spaces around '=' in variable assignments
# Default N = 5, but can be updated by passing "-n $N" in the command-line argument
SUBDIR="./scripts"
N=5
script="slurm_run_ptyrad.sub"

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