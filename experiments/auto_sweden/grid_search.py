import click, time, subprocess

@click.command()
@click.option("-c", "--config_path", type=str, default="suprb2/config.py")
def run_all_experiments(config_path):
    values = values_dictionary()
    config = default_config()

    # Classifier hyperparameters
    for wec in values['weighted_error_const']:
        config['weighted_error_const'] = wec
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
                                for default_error in values['default_error']:
                                    config['default_error'] = default_error
                                    # Rule discovery hyperparameters
                                    for opt in values['rl_name']:
                                        config['rl_name'] = opt
                                        if opt == "'ES_OPL'":
                                            for nrule in values['nrules']:
                                                config['nrules'] = nrule
                                                for rd_steps_per_step in values['rd_steps_per_step']:
                                                    config['rd_steps_per_step'] = rd_steps_per_step
                                                    for lmbd in values['lmbd']:
                                                        config['lmbd'] = lmbd
                                                        for sigma in values['sigma']:
                                                            config['sigma'] = sigma
                                                            start_run(config, config_path)
                                        elif opt == "'ES_MLSP'" or opt == "'ES_CMA'":
                                            for rd_steps_per_step in values['rd_steps_per_step']:
                                                config['rd_steps_per_step'] = rd_steps_per_step
                                                for lmbd in values['lmbd']:
                                                    config['lmbd'] = lmbd
                                                    for mu_denominator in values['mu_denominator']:
                                                        config['mu_denominator'] = mu_denominator
                                                        start_run(config, config_path)
                                        else:
                                            pass


def start_run(config, config_path):
    with open(config_path, "w") as f:
        f.write( get_file_content(config) )
    subprocess.run('sbatch /data/oc-compute01/fischekl/suprb2/slurm/auto_sweden.sbatch')
    time.sleep(300)


def values_dictionary():
    return {
        'rl_name': ["'ES_OPL'", "'ES_MLSP'", "'ES_CMA'", "'ES_ML'"],
        'nrules': [1, 5, 10],
        'lmbd': [10, 25, 35, 50],
        'mu_denominator': [2, 4],       # for 'mu' we are going to use max(lmbd // 'mu_denom', 1)
        'rho_denominator': [1, 2, 4],   # for 'rho' we are going to use max('mu' // 'rho_denom', 2)
        'sigma': [0.01, 0.05, 0.1, 0.15],
        'local_tau': [0.7, 0.9, 1.1, 1.2, 1.3],
        'global_tau': [0.7, 0.9, 1.1, 1.2, 1.3],
        'rd_steps_per_step': [10, 50, 100, 200, 500],
        'recombination': ["None", "'i'", "'d'"],
        'replacement': ["'+'", "','"],
        'start_points': ["'d'", "'u'", "'c'"],
        'weighted_error_const': [10, 25, 50, 100, 150, 200],
        # local model is defined in the experiment itself
        'radius': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mutation_rate': [0.1, 0.2, 0.3, 0.4],
        'fitness': ["'mse'", "'mse_times_C'", "'mse_times_root_C'"],
        'sc_steps_per_step': [10 , 50, 100, 200, 500],
        'initial_pool_size': [50, 75, 100, 150, 200],
        'steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'default_error': [100, 500, 1000, 1500, 2000]
    }


def default_config():
    return {
        'rl_name': "'ES_OPL'", 'nrules': 1, 'lmbd': 10, 'mu_denominator': 2,
        'rho_denominator': 1, 'sigma': 0.01, 'local_tau': 0.7, 'global_tau': 0.7,
        'rd_steps_per_step': 10, 'recombination': "None", 'replacement': "'+'",
        'start_points': "'d'", 'weighted_error_const': 10, 'radius': 0.1,
        'mutation_rate': 0.1, 'fitness': "'mse'", 'sc_steps_per_step': 10,
        'initial_pool_size': 50, 'steps': 1, 'default_error': 100
    }


def get_file_content(config):
    mu = max(config['lmbd'] // config['mu_denominator'], 1)
    rho = max(mu // config['rho_denominator'], 2)
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
            "rho": {rho},
            "sigma": {config['sigma']},
            "local_tau": {config['local_tau']},
            "global_tau": {config['global_tau']},
            "steps_per_step": {config['rd_steps_per_step']},
            "recombination": {config['recombination']},
            "replacement": {config['replacement']},
            "start_points": {config['start_points']}
        }},
        "classifier": {{
            "weighted_error_const": {config['weighted_error_const']},
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
        "default_error": {config['default_error']}
    }}

    def __init__(self):
        self.__dict__ = self.__shared_state

'''


if __name__ == '__main__':
    run_all_experiments()
