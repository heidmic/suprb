import click
import numpy as np

@click.command()
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("-c", "--config_path", type=str, default="suprb2/config.py")
def rewrite_config(seed, config_path):
    # Define a seed, so that we are always creating the same variations
    np.random.seed(seed)
    used_configs = get_used_configs(config_path)
    value_ranges = fetch_value_ranges()
    config = fetch_config_for_experiment(value_ranges, used_configs)
    # Write the new config
    with open(config_path, "w") as f:
        f.write( get_file_content(config) )
    # Write used config as comment in the end of the file
    with open(config_path, "a") as f:
        for config in used_configs:
            f.write( f"# -> {str(config)}\n" )


def get_used_configs(config_path):
    used_configs = list()
    with open(config_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 45 and line.startswith("# -> "):
                used_configs.append( list( map(int, line[6:-2].split(", ")) ) )
    return used_configs


def fetch_config_for_experiment(values_ranges, used_configs):
    config_already_used = True
    while config_already_used:
        indices = [ np.random.choice(np.arange(len(hyperparam_range)), replace=False) for hyperparam_range in values_ranges.values() ]
        config_already_used = indices in used_configs
        if not config_already_used:
            used_configs.append(indices)
    range_array = [ list(values_ranges[k]) for k in values_ranges ]
    return [ range_array[i][indices[i]] for i in range(len(indices)) ]


def fetch_value_ranges():
    return {
        'rl_name': ["'ES_OPL'", "'ES_ML'", "'ES_MLSP'", "'ES_CMA'"],
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
        'sc_steps_per_step': [10 , 50, 100, 200, 500],
        'initial_pool_size': [50, 75, 100, 150, 200],
        'steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'default_error': [100, 500, 1000, 1500, 2000]
    }


def get_file_content(config):
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
            "name": {config[0]},
            "nrules": {config[1]},
            "lmbd": {config[2]},  # not allowed to use lambda
            "mu": {max(config[2] // config[3], 1)},
            "rho": {max(config[3] // config[4], 2)},
            "sigma": {config[5]},
            "local_tau": {config[6]},
            "global_tau": {config[7]},
            "steps_per_step": {config[8]},
            "recombination": {config[9]},
            "replacement": {config[10]},
            "start_points": {config[11]}
        }},
        "classifier": {{
            "weighted_error_const": {config[12]},
            "local_model": 'linear_regression',
            "radius": {config[13]},
        }},
        "solution_creation": {{
            "name": '(1+1)-ES',
            # "pop_size": 1,
            # "crossover_type": None,
            "mutation_rate": {config[14]},
            "fitness": "pseudo-BIC",
            "steps_per_step": {config[15]}
        }},
        "initial_pool_size": {config[16]},
        "initial_genome_length": 100000,
        "steps": {config[17]},
        "use_validation": False,
        "logging": True,
        "default_error": {config[18]}
    }}

    def __init__(self):
        self.__dict__ = self.__shared_state

'''


if __name__ == '__main__':
    rewrite_config()
