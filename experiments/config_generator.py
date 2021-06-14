import click
import numpy as np
import pandas as pd

@click.command()
@click.option("-i", "--index", type=click.IntRange(min=0, max=500), default=0)
def rewrite_config(index):
    with open("suprb2/config.py", "w") as f:
        config = fetch_config_for_experiment(index)
        f.write( get_file_content(config) )


def fetch_config_for_experiment(index):
    df = fetch_config_for_experiment()
    indices = np.random.choice(np.arange(len(df.columns)), len(df), replace=True)
    return df.to_numpy()[np.arange(len(df)), indices]


def fetch_value_ranges():
    return pd.DataFrame({
        'rl_name': ['ES_OPL', 'ES_ML', 'ES_MLSP', 'ES_CMA'],
        'nrules': [1, 5, 10],
        'lmbd': [10, 25, 35, 50],
        'mu_denominator': [2, 4],       # for 'mu' we are going to use lmbd // 'mu_denom'
        'rho_denominator': [1, 2, 4],   # for 'rho' we are going to use 'mu' // 'rho_denom'
        'sigma': [0.01, 0.05, 0.1, 0.15],
        'local_tau': [],
        'global_tau': [],
        'rd_steps_per_step': [],
        'recombination': [None, 'i', 'd'],
        'replacement': ['+', ','],
        'start_points': ['d', 'u', 'c'],
        'weighted_error_const': [10, 25, 50, 100, 150, 200],
        # local model is defined in the experiment itself
        'radius': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mutation_rate': [0.1, 0.2, 0.3, 0.4],
        'sc_steps_per_step': [],
        'initial_pool_size': [50, 75, 100, 150, 200],
        'steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'default_error': [100, 500, 1000, 1500, 2000]
    })


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
