from typing import Type

from src.reward_shaping.entrypoint_creator import (
    HumanoidRewardShapingEntrypointCreator,
    MountainCarContinuousRewardShapingEntrypointCreator,
    RewardShapingEntrypointCreatorBase,
    PyBulletHumanoidRewardShapingEntrypointCreator
)


class RewardShapingEntrypointCreatorFactory:
    ENTRYPOINT_MAPPING = {
        "Humanoid-v2": HumanoidRewardShapingEntrypointCreator,
        "MountainCarContinuous-v0": MountainCarContinuousRewardShapingEntrypointCreator,
        "HumanoidBulletEnv-v0": PyBulletHumanoidRewardShapingEntrypointCreator
    }

    @staticmethod
    def get(name: str) -> Type[RewardShapingEntrypointCreatorBase]:
        entrypoint_creator = RewardShapingEntrypointCreatorFactory.ENTRYPOINT_MAPPING.get(name, None)

        if entrypoint_creator is None:
            raise ValueError(
                f"Unknown environment: {name}, viable options are: {RewardShapingEntrypointCreatorFactory.ENTRYPOINT_MAPPING.keys()}")

        return entrypoint_creator
