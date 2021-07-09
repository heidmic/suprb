#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2/output/output-%A-%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --job-name=multidim-cubic
#SBATCH --array=0-3
job_dir=/data/oc-compute01/fischekl/suprb2
experiment=experiments/multidim_cubic/single_run.py
config_path=suprb2/config.py

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID -k 5 -d 2500 -c '$config_path'"
