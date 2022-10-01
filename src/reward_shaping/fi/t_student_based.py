import numpy as np
from scipy.stats import t

from src.Constants import Constants
from src.reward_shaping.fi.fi_base import Fi


class TStudentHeightLowPenaltyShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state) + 500

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 10 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)
