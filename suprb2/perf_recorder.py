import numpy as np


class PerfRecorder:
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {"elitist_val_error": [],
                      "elitist_f1_score": [],
                      "elitist_fitness": [],
                      "elitist_matched": [],
                      "elitist_complexity": [],
                      "val_size": []}

    def __init__(self):
        self.__dict__ = self.__shared_state
