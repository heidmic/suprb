#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2-multidim-cubic/output/output-%A-%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --job-name=multidim-cubic
job_dir=/data/oc-compute01/fischekl/suprb2-multidim-cubic
experiment=experiments/multidim_cubic/single_run.py

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed 0 -k 5 -d 2500"

