from typing import Union

import numpy as np
from tqdm import tqdm

from suprb import SupRB
from . import BaseLogger


class StdoutLogger(BaseLogger):
    """Print useful statistics to stdout."""

    iterator_: Union[range, tqdm]

    def __init__(self, progress_bar: bool = False):
        self.progress_bar = progress_bar

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: SupRB):
        if self.progress_bar:
            self.iterator_ = tqdm(
                desc="Fitting SupRB", total=estimator.n_iter, ncols=80
            )

    def log_iteration(
        self, X: np.ndarray, y: np.ndarray, estimator: SupRB, iteration: int
    ):

        elitist = estimator.solution_composition_.elitist()

        message = str(
            dict(
                error=elitist.error_,
                fitness=elitist.fitness_,
                complexity=elitist.complexity_,
                score=elitist.score(X, y),
            )
        )

        message = f"[{iteration + 1}/{estimator.n_iter}] Statistics: {message}"

        if self.progress_bar:
            self.iterator_.write(message)
            self.iterator_.update()
        else:
            print(message)

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: SupRB):

        score = estimator.score(X, y)
        message = f"Final training score: {score}"

        if self.progress_bar:
            self.iterator_.write(message)
        else:
            print(message)

    def get_elitist(self, estimator: SupRB):
        pass
