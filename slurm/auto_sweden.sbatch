#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2/output/output-%A-%a.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=500
job_dir=/data/oc-compute01/fischekl/suprb2/
experiment=experiments/auto_sweden/single_run.py

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed 0 --data-seed 0"
