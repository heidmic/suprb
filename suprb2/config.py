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
            "cl_expected_radius": 0.2,
            "nrules": 50,
            "lmbd": 20,  # not allowed to use lambda
            "sigma": 0.2,
            "steps_per_step": 50,
            "weighted_error_constant": 100
        },
        "solution_creation": {
            "name": '(1+1)-ES',
            # "pop_size": 1,
            # "crossover_type": None,
            "mutation_rate": 0.2,
            "fitness": "pseudo-BIC",
            "steps_per_step": 100
        },
        "initial_pool_size": 50,
        "initial_genome_length": 100000,
        "steps": 500,
        "use_validation": False,
        "logging": True,
        "default_error": 1000
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
