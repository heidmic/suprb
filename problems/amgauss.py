import numpy as np  # type: ignore
import scipy as sp  # type: ignore
from problems.gaussiansum import GaussianSumProblem
from problems.problem import Problem


class AMGaussProblem(GaussianSumProblem):
    def __init__(self, seed: int):

        # temp_room
        # humid_room
        # object
        # printer
        # material
        xdim: int = 5
        # speed_print
        # temp_print
        # temp_bed
        # max_speed_fan
        # retraction
        # speed_retraction
        adim: int = 6

        super(AMGaussProblem, self).__init__(seed=seed, xdim=xdim, adim=adim)
