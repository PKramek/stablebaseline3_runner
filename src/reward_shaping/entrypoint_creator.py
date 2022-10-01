from abc import ABC, abstractmethod

from src.reward_shaping.fi.fi_base import Fi
from src.reward_shaping.reward_shaping_creator import RewardShapingEnvironmentCreator


class RewardShapingEntrypointCreatorBase(ABC):
    def __init__(self, shaped_env_name: str, gamma: float, fi: Fi):
        self.shaped_env_name = shaped_env_name
        self._gamma = gamma
        self.fi = fi

    def create(self):
        entrypoint = RewardShapingEnvironmentCreator(
            env=self.base_env_name,
            gamma=self._gamma,
            fi=self.fi,
            fi_t0=self.initial_fi_value
        )

        return entrypoint

    @property
    def shaped_env_name(self) -> str:
        return self._shaped_env_name

    @shaped_env_name.setter
    def shaped_env_name(self, shaped_env_name: str) -> str:
        if not isinstance(shaped_env_name, str):
            raise TypeError("shaped_env_name parameter must be a string")

        self._shaped_env_name = shaped_env_name

    @property
    def fi(self) -> Fi:
        return self._fi

    @fi.setter
    def fi(self, fi: Fi):
        if not isinstance(fi, Fi):
            raise TypeError("fi must be an instance of class Fi")
        self._fi = fi

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        if not isinstance(gamma, float):
            raise TypeError("gamma parameter must be a float")
        self._gamma = gamma

    @property
    @abstractmethod
    def base_env_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def initial_fi_value(self):
        raise NotImplementedError()


class HumanoidRewardShapingEntrypointCreator(RewardShapingEntrypointCreatorBase):

    @property
    def base_env_name(self) -> str:
        return "Humanoid-v2"

    @property
    def initial_fi_value(self) -> float:
        # 1.4 is the value of mean (starting) value of vertical position (z-axis)
        return self._fi([1.4])


# This class is only used for debugging purposes
class MountainCarContinuousRewardShapingEntrypointCreator(RewardShapingEntrypointCreatorBase):
    @property
    def base_env_name(self) -> str:
        return "MountainCarContinuous-v0"

    @property
    def initial_fi_value(self) -> float:
        return self._fi([0])
