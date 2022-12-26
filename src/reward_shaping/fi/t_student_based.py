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

class TStudentHeightLowPenaltyNotShifted(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state)

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 10 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)

class TStudentHeightLowPenaltyBigDifferences(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state)

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 50 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)

class TStudentHeightLowPenaltyMediumDifferences(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state)

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 25 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class JustFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return 500.0

class LinearShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state)

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        return -(np.abs(state[index] - middle_of_dist) * 6) + 501.4


class PyBulletLinearShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state)

    def _base_penalty(self, state: np.ndarray):
        index = Constants.PYBULLET_HEIGHT_INDEX
        middle_of_dist = Constants.PYBULLET_HEIGHT_NOMINAL_VALUE

        return -(np.abs(state[index] - middle_of_dist) * 6) + 501.4


class PyBulletTStudentHeightLowPenaltyShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state) + 500

    def _base_penalty(self, state: np.ndarray):
        index = Constants.PYBULLET_HEIGHT_INDEX
        middle_of_dist = Constants.PYBULLET_HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 10 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)
