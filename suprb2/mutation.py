import numpy as np
from abc import ABC, abstractmethod
from suprb2.random_gen import Random


class MutationStrategies(object):

    def __init__(self, mutation):
        self._mutation = mutation

    def do_mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        self._mutation.mutate_cl(cl, X, y, mutProb, xv, yv, gen, max_gen)
        # return self._mutation.mutate(cl)

    def do_mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        self._mutation.mutate_ind(ind, X, y, mutProb, xv, yv, gen, max_gen)


class Mutation(ABC):

    @abstractmethod
    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        pass

    @abstractmethod
    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        pass


"""Simple Mutation Strategies:
    - GaussianMutation
    - UniformMutation
    - ClassifierStochasticHCLike
"""


class GaussianMutation(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
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
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates this matching function.

        This is done similar to how the first XCSF iteration used mutation
        (Wilson, 2002) but using a Gaussian distribution instead of a uniform
        one (as done by Drugowitsch, 2007): Each interval [l, u)'s bound x is
        changed to x' ~ N(x, (u - l) / 10) (Gaussian with standard deviation a
        10th of the interval's width).
        """
        for cl in ind.classifiers:
            if Random().random.random() < mutProb:
                lowers = Random().random.normal(loc=cl.lowerBounds, scale=2/10, size=len(cl.lowerBounds))
                uppers = Random().random.normal(loc=cl.upperBounds, scale=2/10, size=len(cl.upperBounds))
                lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                cl.lowerBounds = lu[0]
                cl.upperBounds = lu[1]


class UniformMutation(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using a uniform distribution over the interval [-1, 1).
        Does not produce solutions better than initial elitist with mutProb = 1.0.
        Might be useful to explore search space.
        Solution quality strongly dependent on mutProb.
        """
        lowers = Random().random.uniform(low=-1.0, high=1.0, size=len(cl.lowerBounds))
        uppers = Random().random.uniform(low=-1.0, high=1.0, size=len(cl.upperBounds))
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using a uniform distribution over the interval [-1, 1).
        Does not produce solutions better than initial elitist with mutProb = 1.0.
        Might be useful to explore search space.
        Solution quality strongly dependent on mutProb.
        """
        for cl in ind.classifiers:
            if Random().random.random() < mutProb:
                lowers = Random().random.uniform(low=-1.0, high=1.0, size=len(cl.lowerBounds))
                uppers = Random().random.uniform(low=-1.0, high=1.0, size=len(cl.upperBounds))
                lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                cl.lowerBounds = lu[0]
                cl.upperBounds = lu[1]


class ClassifierStochasticHCLike(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and the best move is selected.
        Presumably causes insufficient matching when used alone; does however produce better classifier and can
        be used in combination with other mutation strategies.
        """
        if Random().random.random() < mutProb:
            moves = 20
            if cl.error is not None:
                new_bounds = np.array([cl.lowerBounds, cl.upperBounds])
                minerror = cl.error
                initialLowers = cl.lowerBounds
                initialUppers = cl.upperBounds
                for i in range(moves):
                    lowers = Random().random.normal(loc=cl.lowerBounds, scale=2 / 10, size=len(cl.lowerBounds))
                    uppers = Random().random.normal(loc=cl.upperBounds, scale=2 / 10, size=len(cl.upperBounds))
                    lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                    cl.lowerBounds = lu[0]
                    cl.upperBounds = lu[1]
                    cl.fit(X, y)
                    cl.lowerBounds = initialLowers
                    cl.upperBounds = initialUppers
                    if cl.error < minerror:
                        new_bounds[0] = lu[0]
                        new_bounds[1] = lu[1]
                        minerror = cl.error
                cl.lowerBounds = new_bounds[0]
                cl.upperBounds = new_bounds[1]
                cl.fit(X, y)

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates several classifiers within an individual.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and the best move is selected.
        Presumably causes insufficient matching when used alone; does however produce better classifier and can
        be used in combination with other mutation strategies.
        """
        for cl in ind.classifiers:
            if Random().random.random() < mutProb:
                moves = 20
                if cl.error is not None:
                    new_bounds = np.array([cl.lowerBounds, cl.upperBounds])
                    minerror = cl.error
                    initialLowers = cl.lowerBounds
                    initialUppers = cl.upperBounds
                    for i in range(moves):
                        lowers = Random().random.normal(loc=cl.lowerBounds, scale=2 / 10, size=len(cl.lowerBounds))
                        uppers = Random().random.normal(loc=cl.upperBounds, scale=2 / 10, size=len(cl.upperBounds))
                        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                        cl.lowerBounds = lu[0]
                        cl.upperBounds = lu[1]
                        cl.fit(X, y)
                        cl.lowerBounds = initialLowers
                        cl.upperBounds = initialUppers
                        if cl.error < minerror:
                            new_bounds[0] = lu[0]
                            new_bounds[1] = lu[1]
                            minerror = cl.error
                    cl.lowerBounds = new_bounds[0]
                    cl.upperBounds = new_bounds[1]
                    cl.fit(X, y)


class NonUniformMutation(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using the non-uniform mutation as described in De Jong: Handbook of Evolutionary Computation.
        The non-uniformity is dependent on the current generation in relation to the maximum generations.
        b determines degree of non-uniformity.

        Other descriptions use u ** ((1-gen/gem_max) ** b)
        """
        b = 2
        u = Random().random.uniform(low=0.0, high=1.0, size=1)
        if u < 0.5:
            delta_lowers = (1.0 - cl.lowerBounds) * u * (1.0 - gen/max_gen) ** b
            delta_uppers = (1.0 - cl.upperBounds) * u * (1.0 - gen / max_gen) ** b
            lowers = cl.lowerBounds + delta_lowers
            uppers = cl.upperBounds + delta_uppers
        else:
            delta_lowers = (cl.lowerBounds - 1.0) * u * (1.0 - gen / max_gen) ** b
            delta_uppers = (cl.upperBounds - 1.0) * u * (1.0 - gen / max_gen) ** b
            lowers = cl.lowerBounds + delta_lowers
            uppers = cl.upperBounds + delta_uppers
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using the non-uniform mutation as described in De Jong: Handbook of Evolutionary Computation.
        The non-uniformity is dependent on the current generation in relation to the maximum generations.
        b determines degree of non-uniformity.

        Other descriptions use u ** ((1-gen/gem_max) ** b)
        """
        for cl in ind.classifiers:
            if Random().random.random() < mutProb:
                b = 3
                u = Random().random.uniform(low=0.0, high=1.0, size=1)
                if u < 0.5:
                    delta_lowers = (1.0 - cl.lowerBounds) * u * (1.0 - gen / max_gen) ** b
                    delta_uppers = (1.0 - cl.upperBounds) * u * (1.0 - gen / max_gen) ** b
                    lowers = cl.lowerBounds + delta_lowers
                    uppers = cl.upperBounds + delta_uppers
                else:
                    delta_lowers = (cl.lowerBounds - 1.0) * u * (1.0 - gen / max_gen) ** b
                    delta_uppers = (cl.upperBounds - 1.0) * u * (1.0 - gen / max_gen) ** b
                    lowers = cl.lowerBounds + delta_lowers
                    uppers = cl.upperBounds + delta_uppers
                lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                cl.lowerBounds = lu[0]
                cl.upperBounds = lu[1]

# TODO test variation of non-uniform mutation mutating either upper or lower bounds


"""Complex/Experimental Mutation Strategies:
    - IndividualStochasticHCLike
    - StochasticHillClimbing
    - DirectedMutationLinExtrapolation
Do require additional information/function evaluations! Computational expensive! 
Benefit presumably only in special cases!
"""


class IndividualStochasticHCLike(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and
        the probability of choosing a move varies with the amount of improvement of the individual.
        """
        # TODO does not exist for individual fitness

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates several classifiers within an individual.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and
        the probability of choosing a move varies with the amount of improvement of the individual.

        Extremely slow due to repeated fitness determination of individuals!!!
        """

        for cl in ind.classifiers:
            if Random().random.random() < mutProb:
                moves = 20
                ind.fit(X, y)
                ind.determine_fitness(xv, yv)
                if ind.error is not None:
                    new_bounds = np.array([cl.lowerBounds, cl.upperBounds])
                    minerror = ind.error
                    initialLowers = cl.lowerBounds
                    initialUppers = cl.upperBounds
                    for i in range(moves):
                        lowers = Random().random.normal(loc=cl.lowerBounds, scale=2 / 10, size=len(cl.lowerBounds))
                        uppers = Random().random.normal(loc=cl.upperBounds, scale=2 / 10, size=len(cl.upperBounds))
                        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
                        cl.lowerBounds = lu[0]
                        cl.upperBounds = lu[1]
                        ind.determine_fitness(xv, yv)
                        cl.lowerBounds = initialLowers
                        cl.upperBounds = initialUppers
                        if ind.error < minerror:
                            new_bounds[0] = lu[0]
                            new_bounds[1] = lu[1]
                            minerror = ind.error
                    cl.lowerBounds = new_bounds[0]
                    cl.upperBounds = new_bounds[1]
                    ind.determine_fitness(xv, yv)


class StochasticHillClimbing(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and
        the probability of choosing a move varies with the amount of improvement.
        """

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates several classifiers within an individual.

        This is done using a method related to stochastic hill climbing.
        Several moves are computed by Gaussian Mutation and
        the probability of choosing a move varies with the amount of improvement.
        """


# TODO DOES NOT WORK YET
class DirectedMutationLinExtrapolation(Mutation):

    def mutate_cl(self, cl, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates a single classifier.

        This is done similar to the mutation proposed by Bhandari et al. (1994). It uses linear
        extrapolation between the current solution and the best previous solution.
        Possible Modifications: use best current solution
        """
        # alpha is factor to determine the step size towards the best solution; problem dependent, 0 < alpha < 1
        # 0.2 on average best alpha
        # TODO reimplement best values using neighbourhood definition
        alpha = 0.4
        lowers = cl.lowerBounds + alpha * (cl.lowerBounds - cl.bestLower)
        uppers = cl.upperBounds + alpha * (cl.upperBounds - cl.bestUpper)
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        cl.lowerBounds = lu[0]
        cl.upperBounds = lu[1]
        # return lu[0], lu[1]

    def mutate_ind(self, ind, X, y, mutProb, xv, yv, gen, max_gen):
        """
        Mutates several classifiers within an individual.

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

