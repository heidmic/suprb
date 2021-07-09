import click, time, subprocess

@click.command()
@click.option("-c", "--config_path", type=str, default="suprb2/config.py")
def run_all_experiments(config_path):
    values = values_dictionary()
    config = default_config()

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
                                        for opt in values['rl_name']:
                                            config['rl_name'] = opt

                                            if opt == "'ES_OPL'":
                                                for nrule in values['nrules']:
                                                    config['nrules'] = nrule
                                                    for sigma in values['sigma']:
                                                        config['sigma'] = sigma
                                                        start_run(config, config_path)
                                            elif opt == "'ES_MLSP'":
                                                for mu_denominator in values['mu_denominator_MLSP']:
                                                    config['mu_denominator'] = mu_denominator
                                                    start_run(config, config_path)
                                            elif opt == "'ES_CMA'":
                                                for mu_denominator in values['mu_denominator_CMA']:
                                                    config['mu_denominator'] = mu_denominator
                                                    start_run(config, config_path)
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
                                                                        start_run(config, config_path)


def values_dictionary():
    return {
        # Rule Discovery
        'rl_name': ["'ES_ML'", "'ES_OPL'", "'ES_MLSP'", "'ES_CMA'"],
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
        'fitness': ["'pseudo-BIC'", "'inverted_macro_f1_score'", "'inverted_macro_f1_score_times_C'"],
        'sc_steps_per_step': [100],
        # LCS
        'initial_pool_size': [500],
        'steps': [100]
    }


def start_run(config, config_path):
    with open(config_path, "w") as f:
        f.write( get_file_content(config) )
    proc = subprocess.Popen(['sbatch /data/oc-compute01/fischekl/suprb2-haberman/slurm/haberman.sbatch'], shell=True)
    proc.wait()
    time.sleep(150)


def default_config():
    return {
        'rl_name': "'ES_OPL'", 'nrules': 1, 'lmbd': 10, 'mu_denominator': 2,
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
