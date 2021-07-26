class Config:
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {
        "n_elitists": 1,
        "rule_discovery": {
            "name": 'ES_ML',
            "nrules": 1,
            "lmbd": 32,  # not allowed to use lambda
            "mu": 4,
            "rho": 2,
            "sigma": 0.01,
            "local_tau": 1.1,
            "global_tau": 1.2,
            "steps_per_step": 100,
            "recombination": None,
            "replacement": '+',
            "start_points": 'c'
        },
        "classifier": {
            "weighted_error_const": 0.5,
            "local_model": 'logistic_regression',
            "radius": 0.1,
        },
        "solution_creation": {
            "name": '(1+1)-ES',
            # "pop_size": 1,
            # "crossover_type": None,
            "mutation_rate": 0.1,
            "fitness": 'inverted_macro_f1_score_times_C',
            "fitness_target": 1e-3,
            "fitness_factor": 2,
            "steps_per_step": 100
        },
        "initial_pool_size": 500,
        "initial_genome_length": 100000,
        "steps": 100,
        "use_validation": False,
        "logging": True,
        "default_error": 1000
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
