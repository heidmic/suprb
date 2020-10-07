from .amgauss import *
from .problem import *

classes = {
    "amgauss": AMGaussProblem
}


def make_problem(name: str, seed: int, *args) -> Problem:
    return classes[name](seed=seed, *args)
