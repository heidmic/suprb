#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2/output/output-%A-%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --job-name=sonar
#SBATCH --array=0-19
job_dir=/data/oc-compute01/fischekl/suprb2
experiment=experiments/sonar/single_run.py

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID --data-seed 0 --run-name ML_time_final --config-path configs/classification/ML_time_final_config.py"
