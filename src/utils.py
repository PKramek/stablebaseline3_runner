import string
from datetime import datetime
from random import choice as random_choice

import gym
from stable_baselines3.common.utils import set_random_seed

from src.Constants import Constants


def get_tensorboard_logs_directory(algorithm: str) -> str:
    return f"{Constants.LOGS_DIRECTORY}/{algorithm}"


def get_random_id(length: int = 6):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random_choice(chars) for _ in range(length))


def get_run_name(algorithm: str, environment: str) -> str:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H_%M_%S")
    random_id = get_random_id(length=Constants.RANDOM_ID_LENGTH)

    return f"{environment}_{algorithm}_{random_id}_{timestamp}"


def get_experiment_tb_directory(tensorboard_logs_directory: str, run_name: str):
    return f"{tensorboard_logs_directory}/{run_name}"


