import numpy as np

from suprb.rule import Rule
from suprb.utils import RandomState
from . import MixingModel


class ErrorExperienceHeuristicOld(MixingModel):
    """
    Performs mixing similar to the Inverse Variance Heuristic from
    https://researchportal.bath.ac.uk/en/studentTheses/learning-classifier-systems-from-first-principles-a-probabilistic,
    but using (error / experience) as a mixing function.
    """

    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        input_size = X.shape[0]

        # No need to perform any calculation if no rule was selected.
        if not subpopulation:
            return np.zeros(input_size)

        # Get errors and experience of all rules in subpopulation
        experiences = np.array([rule.experience_ for rule in subpopulation])
        errors = np.array([rule.error_ for rule in subpopulation])

        local_pred = np.zeros((len(subpopulation), input_size))

        if cache:
            # Use the precalculated matches and predictions from fit()
            matches = [rule.match_set_ for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                local_pred[i][matches[i]] = rule.pred_
        else:
            # Generate all data new
            matches = [rule.match(X) for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                if not matches[i].any():
                    continue
                local_pred[i][matches[i]] = rule.predict(X[matches[i]])

        taus = (1 / errors) * experiences

        # Stack all local predictions and sum them weighted with tau
        pred = np.sum(local_pred * taus[:, None], axis=0)

        # Sum all taus
        local_taus = np.zeros((len(subpopulation), input_size))
        for i in range(len(subpopulation)):
            local_taus[i][matches[i]] = taus[i]

        tau_sum = np.sum(local_taus, axis=0)
        tau_sum[tau_sum == 0] = 1  # Needed

        # Normalize
        out = pred / tau_sum
        return out

class ErrorExperienceHeuristic(MixingModel):
    
    def __init__(self, random_state: RandomState, rule_amount: int = 4):
        self.random_state = random_state
        self.rule_amount = rule_amount

    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        input_size = X.shape[0]

        # No need to perform any calculation if no rule was selected.
        if not subpopulation:
            return np.zeros(input_size)
        
        ###############################################################

        # Filters out rule_amount best rules and only uses that for mixing
        fitnesses = np.array([rule.fitness_ for rule in subpopulation])
        ind = np.argpartition(fitnesses, -self.rule_amount)[-self.rule_amount:]
        subpopulation = subpopulation[ind]

        # Same for random
        subpopulation = self.random_state.sample(subpopulation, self.rule_amount)

        # RouletteWheel
        fitnesses = np.array([rule.fitness_ for rule in subpopulation])
        weights = fitnesses / np.sum(fitnesses)
        subpopulation = self.random_state.choice(subpopulation, p=weights, size=self.rule_amount)

        ###############################################################


        # Get errors and experience of all rules in subpopulation
        experiences = np.array([rule.experience_ for rule in subpopulation])
        errors = np.array([rule.error_ for rule in subpopulation])

        local_pred = np.zeros((len(subpopulation), input_size))

        if cache:
            # Use the precalculated matches and predictions from fit()
            matches = [rule.match_set_ for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                local_pred[i][matches[i]] = rule.pred_
        else:
            # Generate all data new
            matches = [rule.match(X) for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                if not matches[i].any():
                    continue
                local_pred[i][matches[i]] = rule.predict(X[matches[i]])

        taus = (1 / errors) * experiences

        # Stack all local predictions and sum them weighted with tau
        pred = np.sum(local_pred * taus[:, None], axis=0)

        # Sum all taus
        local_taus = np.zeros((len(subpopulation), input_size))
        for i in range(len(subpopulation)):
            local_taus[i][matches[i]] = taus[i]

        tau_sum = np.sum(local_taus, axis=0)
        tau_sum[tau_sum == 0] = 1  # Needed

        # Normalize
        out = pred / tau_sum
        return out
