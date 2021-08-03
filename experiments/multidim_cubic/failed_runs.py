from shutil import copyfile
import subprocess

def run_failed_experiments():
    with open('failed-jobs.txt') as jobs_list_file:
        for line in jobs_list_file:
            job_id = line[:5]
            config_path = get_config_path(job_id)
            if config_path is None:
                print(f"Couldn't find config path for job: {job_id}")
            else:
                reschedule_run(config_path)


def get_config_path(job_id):
    config_path = None
    with open(f"output/output-{job_id}-0.txt") as output_file:
        for line in output_file:
            if line[:26] == "Configurations directory: ":
                config_path = line[26:].strip()
                break
    return config_path


def reschedule_run(config_path):
    write_batch(config_path)
    proc = subprocess.Popen(['sbatch /data/oc-compute01/fischekl/suprb2/slurm/multidim_cubic.sbatch'], shell=True)
    proc.wait()


def write_batch(config_path):
    with open("/data/oc-compute01/fischekl/suprb2/slurm/multidim_cubic.sbatch", "w") as f:
        f.write( sbatch_content(config_path) )


def sbatch_content(config_path):
    return \
f'''#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2/output/retry-output-%A-%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --job-name=mdc-failed-ml
#SBATCH --array=0-3
job_dir=/data/oc-compute01/fischekl/suprb2
experiment=experiments/multidim_cubic/single_run.py
config_path={config_path}

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID -k 5 -d 2500 -c '$config_path'"
'''


if __name__ == '__main__':
    run_failed_experiments()
