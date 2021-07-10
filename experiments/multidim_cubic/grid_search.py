import click, subprocess

@click.command()
@click.option("-o", "--optimizer", type=str)
@click.option("-c", "--config_path", type=str, default="suprb2")
def run_all_experiments(optimizer, config_path):
    if optimizer not in ["ES_ML", "ES_OPL", "ES_CMA", "ES_MLSP"]:
        print("Undefined or unknown rule discovery optimizer. Shutting down...")
        return

    values = values_dictionary()
    config = default_config(optimizer)
    iterations = 0

    # Classifier hyperparameters
    for radius in values['radius']:
        config['radius'] = radius
        # Solution Creation hyperparameters
        for mutation_rate in values['mutation_rate']:
            config['mutation_rate'] = mutation_rate
            for fitness in values['fitness']:
                config['fitness'] = fitness
                for sc_steps_per_step in values['sc_steps_per_step']:
                    config['sc_steps_per_step'] = sc_steps_per_step
                    # General hyperparameters
                    for initial_pool_size in values['initial_pool_size']:
                        config['initial_pool_size'] = initial_pool_size
                        for steps in values['steps']:
                            config['steps'] = steps
                            # Rule discovery hyperparameters
                            for rd_steps_per_step in values['rd_steps_per_step']:
                                config['rd_steps_per_step'] = rd_steps_per_step
                                for lmbd in values['lmbd']:
                                    config['lmbd'] = lmbd
                                    for start_points in values['start_points']:
                                        config['start_points'] = start_points

                                        if optimizer == "ES_OPL":
                                            for nrule in values['nrules']:
                                                config['nrules'] = nrule
                                                for sigma in values['sigma']:
                                                    config['sigma'] = sigma
                                                    iterations = start_run(config, config_path, iterations)
                                        elif optimizer == "ES_MLSP":
                                            for mu_denominator in values['mu_denominator_MLSP']:
                                                config['mu_denominator'] = mu_denominator
                                                iterations = start_run(config, config_path, iterations)
                                        elif optimizer == "ES_CMA":
                                            for mu_denominator in values['mu_denominator_CMA']:
                                                config['mu_denominator'] = mu_denominator
                                                iterations = start_run(config, config_path, iterations)
                                        else:
                                            for mu_denominator in values['mu_denominator_ML']:
                                                config['mu_denominator'] = mu_denominator
                                                for rho in unique_rho_values(values['rho_denominator'], mu_denominator, lmbd):
                                                    config['rho'] = rho
                                                    for recombination in values['recombination']:
                                                        config['recombination'] = recombination
                                                        for replacement in values['replacement']:
                                                            config['replacement'] = replacement
                                                            for local_tau in values['local_tau']:
                                                                config['local_tau'] = local_tau
                                                                for global_tau in values['global_tau']:
                                                                    config['global_tau'] = global_tau
                                                                    iterations = start_run(config, config_path, iterations)


def values_dictionary():
    return {
        # Rule Discovery
        'nrules': [200],
        'lmbd': [8, 16, 32, 64],
        # for 'mu' we are going to use max(lmbd // 'mu_denom', 1)
        'mu_denominator_ML': [7],
        'mu_denominator_CMA': [2],
        'mu_denominator_MLSP': [4],
        'rho_denominator': [1, 2, 4],   # for 'rho' look at unique_rho_values
        'sigma': [0.01, 0.1, 0.2],
        'local_tau': [1.1, 1.2],
        'global_tau': [1.1, 1.2],
        'rd_steps_per_step': [100],
        'recombination': ["None", "'i'", "'d'"],
        'replacement': ["'+'", "','"],
        'start_points': ["'d'", "'u'", "'c'"],
        # Classifiers
        # local model is defined in the experiment itself
        'radius': [0.1, 0.3, 0.5],
        # Solution Creation
        'mutation_rate': [0.1, 0.2, 0.3, 0.4],
        'fitness': ["'mse_times_C'", "'mse_times_root_C'"],
        'sc_steps_per_step': [100],
        # LCS
        'initial_pool_size': [500],
        'steps': [100]
    }


def start_run(config, config_path, iterations):
    create_and_link_config(config_path, config, iterations)
    proc = subprocess.Popen(['sbatch /data/oc-compute01/fischekl/suprb2/slurm/multidim_cubic.sbatch'], shell=True)
    proc.wait()
    return iterations + 1


def create_and_link_config(config_path, config, iterations):
    config_file_path = f"{config_path}/config_{iterations}.py"
    with open(config_file_path, "w") as f:
        f.write( get_file_content(config) )
    with open("/data/oc-compute01/fischekl/suprb2/slurm/multidim_cubic.sbatch", "w") as f:
        f.write( sbatch_content(config_file_path) )


def default_config(optimizer):
    return {
        'rl_name': f"'{optimizer}'", 'nrules': 1, 'lmbd': 10, 'mu_denominator': 2,
        'rho': 1, 'sigma': 0.01, 'local_tau': 0.7, 'global_tau': 0.7,
        'rd_steps_per_step': 10, 'recombination': "None", 'replacement': "'+'",
        'start_points': "'d'", 'radius': 0.1, 'mutation_rate': 0.1,
        'fitness': "'mse'", 'sc_steps_per_step': 10, 'initial_pool_size': 50,
        'steps': 1
    }


def unique_rho_values(rho_denominators, mu_denominator, lmbd):
    mu = max(lmbd // mu_denominator, 1)
    # transform into set, so that we get only unique values, and then turn it back to list
    return list(set([ max(mu // rho_den, 1) for rho_den in rho_denominators ]))


def sbatch_content(config_path):
    return \
f'''#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute01/fischekl/suprb2/output/output-%A-%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500
#SBATCH --job-name=multidim-cubic
#SBATCH --array=0-3
job_dir=/data/oc-compute01/fischekl/suprb2
experiment=experiments/multidim_cubic/single_run.py
config_path={config_path}

srun nix-shell "$job_dir"/slurm/default.nix --command "PYTHONPATH=$job_dir/$PYTHONPATH python $job_dir/$experiment --seed $SLURM_ARRAY_TASK_ID -k 5 -d 2500 -c '$config_path'"
'''


def get_file_content(config):
    mu = max(config['lmbd'] // config['mu_denominator'], 1)
    return \
f'''class Config:
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {{
        "n_elitists": 1,
        "rule_discovery": {{
            "name": {config['rl_name']},
            "nrules": {config['nrules']},
            "lmbd": {config['lmbd']},  # not allowed to use lambda
            "mu": {mu},
            "rho": {config['rho']},
            "sigma": {config['sigma']},
            "local_tau": {config['local_tau']},
            "global_tau": {config['global_tau']},
            "steps_per_step": {config['rd_steps_per_step']},
            "recombination": {config['recombination']},
            "replacement": {config['replacement']},
            "start_points": {config['start_points']}
        }},
        "classifier": {{
            "weighted_error_const": 0.5,
            "local_model": 'linear_regression',
            "radius": {config['radius']},
        }},
        "solution_creation": {{
            "name": '(1+1)-ES',
            # "pop_size": 1,
            # "crossover_type": None,
            "mutation_rate": {config['mutation_rate']},
            "fitness": {config['fitness']},
            "fitness_target": 1e-3,
            "fitness_factor": 2,
            "steps_per_step": {config['sc_steps_per_step']}
        }},
        "initial_pool_size": {config['initial_pool_size']},
        "initial_genome_length": 100000,
        "steps": {config['steps']},
        "use_validation": False,
        "logging": True,
        "default_error": 1000
    }}

    def __init__(self):
        self.__dict__ = self.__shared_state

'''


if __name__ == '__main__':
    run_all_experiments()
