import numpy as np
from abc import ABC, abstractmethod
from suprb2.random_gen import Random


class MutationStrategies(object):

    def __init__(self, mutation):
        self._mutation = mutation

    def do_mutate_cl(self, cl):
        self._mutation.mutate_cl(cl)
        # return self._mutation.mutate(cl)

    def do_mutate_ind(self, ind):
        self._mutation.mutate_ind(ind)


class Mutation(ABC):

    @abstractmethod
    def mutate_cl(self, cl):
        pass

    @abstractmethod
    def mutate_ind(self, ind):
        pass


class GaussianMutation(Mutation):

    def mutate_cl(self, cl):
        """
        Mutates this matching function.

        This is done similar to how the first XCSF iteration used mutation
        (Wilson, 2002) but using a Gaussian distribution instead of a uniform
        one (as done by Drugowitsch, 2007): Each interval [l, u)'s bound x is
        changed to x' ~ N(x, (u - l) / 10) (Gaussian with standard deviation a
        10th of the interval's width).
        """
        lowers = Random().random.normal(loc=cl.lowerBounds, scale=2/10, size=len(cl.lowerBounds))
        uppers = Random().random.normal(loc=cl.upperBounds, scale=2/10, size=len(cl.upperBounds))
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        # return lu[0], lu[1]
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]

    def mutate_ind(self, ind):
        """
        Mutates this matching function.

        This is done similar to how the first XCSF iteration used mutation
        (Wilson, 2002) but using a Gaussian distribution instead of a uniform
        one (as done by Drugowitsch, 2007): Each interval [l, u)'s bound x is
        changed to x' ~ N(x, (u - l) / 10) (Gaussian with standard deviation a
        10th of the interval's width).
        """
        for cl in ind.classifiers:
            lowers = Random().random.normal(loc=cl.lowerBounds, scale=2/10, size=len(cl.lowerBounds))
            uppers = Random().random.normal(loc=cl.upperBounds, scale=2/10, size=len(cl.upperBounds))
            lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
            # return lu[0], lu[1]
            cl.lowerBounds = lu[0]
            cl.upperBounds = lu[1]


# TODO DOES NOT WORK YET
class DirectedMutationLinExtrapolation(Mutation):

    def mutate_cl(self, cl):
        """
        Mutates this matching function.

        This is done similar to the mutation proposed by Bhandari et al. (1994). It uses linear
        extrapolation between the current solution and the best previous solution.
        Possible Modifications: use best current solution
        """
        # alpha is factor to determine the step size towards the best solution; problem dependent, 0 < alpha < 1
        # 0.2 on average best alpha
        alpha = 0.4
        lowers = cl.lowerBounds + alpha * (cl.lowerBounds - cl.bestLower)
        uppers = cl.upperBounds + alpha * (cl.upperBounds - cl.bestUpper)
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]
        # return lu[0], lu[1]

    def mutate_ind(self, ind):
        """
        Mutates this matching function.

        This is done similar to the mutation proposed by Bhandari et al. (1994). It uses linear
        extrapolation between the current solution and the best previous solution.
        Possible Modifications: use best current solution
        """
        # TODO maybe somehow use elitist
        # TODO can not use best values from this classifier if they are not changed
        for cl in ind.classifiers:
            # alpha is factor to determine the step size towards the best solution; problem dependent, 0 < alpha < 1
            # 0.2 on average best alpha
            alpha = 0.8
            lowers = cl.lowerBounds + alpha * (cl.lowerBounds - cl.bestLower)
            uppers = cl.upperBounds + alpha * (cl.upperBounds - cl.bestUpper)
            lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
            cl.lowerBounds = lu[0]
            cl.upperBounds = lu[1]
