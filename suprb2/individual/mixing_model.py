import numpy as np

from suprb2.rule import Rule
from . import MixingModel


class ErrorExperienceHeuristic(MixingModel):
    """
    Performs mixing similar to the Inverse Variance Heuristic from
    https://researchportal.bath.ac.uk/en/studentTheses/learning-classifier-systems-from-first-principles-a-probabilistic,
    but using (error / experience) as a mixing function.
    """

    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        input_size = X.shape[0]

        # No need to perform any calculation if no rule was selected.
        if not subpopulation:
            return np.zeros(X.shape[0])

        # Get errors and experience of all rules in subpopulation
        experiences = np.array([rule.experience_ for rule in subpopulation])
        errors = np.array([rule.error_ for rule in subpopulation])

        local_pred = np.zeros((len(subpopulation), input_size))

        if cache:
            # Use the precalculated matches and predictions from fit()
            matches = [rule.match_ for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                local_pred[i][matches[i]] = rule.pred_
        else:
            # Generate all data new
            matches = [rule.matched_data(X) for rule in subpopulation]
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