class Config:
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {
        "n_elitists": 1,
        "rule_discovery": {
            "name": '(1+lambda)-ES',
            "cl_min_range": 0.2,
            "nrules": 100,
            "lmbd": 20,  # not allowed to use lambda
            "mu": 10,
            "rho": 2,
            "sigma": 0.2,
            "start_points": None,
            "steps_per_step": 10,
            "recombination": None,
            "replacement": '+',
            "min_sigma": 0.8,
            "max_sigma": 1.2,
            "weighted_error_constant": 1000
        },
        "solution_creation": {
            "name": '(1+1)-ES',
            # "pop_size": 1,
            # "crossover_type": None,
            "mutation_rate": 0.2,
            "fitness": "pseudo-BIC",
            "steps_per_step": 1000
        },
        "initial_pool_size": 50,
        "initial_genome_length": 100000,
        "steps": 500,
        "xdim": None,
        "use_validation": False,
        "default_prediction": None,
        "var": None,
        "logging": True
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
