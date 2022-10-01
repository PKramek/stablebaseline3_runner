import logging

import gym

from src.Constants import Constants
from src.reward_shaping.wrapper import RewardShapingWrapper

logger = logging.getLogger(Constants.LOGGER_NAME)


class RewardShapingEnvironmentCreator:
    def __init__(self, *, env: str, gamma: float, fi: callable, fi_t0: float):
        assert isinstance(env, str), "Environment parameter must be a string"
        assert isinstance(gamma, float) and 0 < gamma < 1, "Gamma parameter must be a float in range (0,1)"
        assert callable(fi), "Fi must be a callable"
        assert isinstance(fi_t0, float), f"Fi(t0) (passed value = {fi_t0}) must be a float not {type(fi_t0)}"

        self._environment_name = env
        self._gamma = gamma
        self._fi = fi
        self._fi_t0 = fi_t0

    def _build_env(self):
        env = gym.make(self._environment_name)
        wrapped_env = RewardShapingWrapper(env, self._gamma, self._fi, self._fi_t0)

        return wrapped_env

    def __call__(self, *args, **kwargs):
        return self._build_env()



