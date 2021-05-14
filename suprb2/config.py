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
            "cl_expected_radius": 0.5,
            "nrules": 100,
            "lmbd": 40,  # not allowed to use lambda
            "sigma": 0.1,
            "steps_per_step": 10,
            "weighted_error_constant": 0.5
        },
        "solution_creation": {
            "name": "NSGA-II",
            "steps_per_step": 100,
            "pop_size": 40,
            "recom_prob": 0.5,
            "recom_rate": 0.2,
            "mut_rate": 0.2
        },
        "initial_pool_size": 500,
        "initial_genome_length": 100000,
        "steps": 150,
        "use_validation": False,
        "logging": True,
        "default_error": 1000
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
