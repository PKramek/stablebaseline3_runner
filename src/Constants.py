import torch
from stable_baselines3 import SAC, PPO


class Constants:
    ALGORITHMS = {'SB_PPO': PPO, 'SB_SAC': SAC}

    ACTIVATION_FUNCTIONS_MAPPING = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh
    }

    ENVIRONMENTS_WITH_REWARD_SHAPING = [
        "Humanoid-v2",
        "HumanoidBulletEnv-v0",
        "MountainCarContinuous-v0",
    ]

    RANDOM_ID_LENGTH = 6

    # Env variables used in evaluation results data extraction
    ENV_PROGRESS_FILE_PATH = 'PROGRESS_FILE_PATH'
    ENV_EVALUATION_RESULTS_DIR_PATH = 'EVALUATION_RESULTS_FILE_PATH'
    LOGS_DIRECTORY = '/tensorboard_logs'

    LOG_FORMAT = '%(asctime)s - %(message)s'
    LOGGER_NAME = 'stablebaselines3_runner_logger'

    # Indexes for values in Humanoid-v2 state vector
    HEIGHT_INDEX = 0
    TILT_INDEX = 3  # In qpos its under index 5, but observation cuts first two elements
    X_AXIS_ROTATION_INDEX = 4  # In qpos its under index 6, but observation cuts first two elements
    PYBULLET_HEIGHT_INDEX = 0

    HEIGHT_NOMINAL_VALUE = 1.4
    TILT_NOMINAL_VALUE = 0.1
    X_AXIS_ROTATION_NOMINAL_VALUE = 0.0

    PYBULLET_HEIGHT_NOMINAL_VALUE = 0.6
