#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute02/heidermi/suprb2/output/output-%A-%a.txt
#SBATCH --array=0-19
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1000
job_dir=/data/oc-compute02/heidermi/suprb2/
experiment=experiments/multidim_cubic.py
nix-shell "$job_dir"/default.nix --command "PYTHONPATH=$job_dir/src:$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID --k=2"
