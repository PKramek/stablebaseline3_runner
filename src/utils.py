import string
from datetime import datetime
from random import choice as random_choice

from src.Constants import Constants


def get_tensorboard_logs_directory(algorithm: str) -> str:
    return f"{Constants.LOGS_DIRECTORY}/{algorithm}/"


def get_run_name(algorithm: str, environment: str) -> str:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H_%M_%S")
    random_id = get_random_id(length=Constants.RANDOM_ID_LENGTH)

    return f"{environment}_{algorithm}_{random_id}_{timestamp}"


def get_random_id(chars=string.ascii_uppercase + string.digits, length: int = 6):
    return ''.join(random_choice(chars) for _ in range(length))
