#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute02/heidermi/suprb2/output/output-%A-%a.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=500
#SBATCH --array=0-19
job_dir=/data/oc-compute02/heidermi/suprb2/
experiment=experiments/multidim_cubic.py
srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID -k 10 -d 2500"
