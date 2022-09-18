from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm.auto import tqdm

from src.Constants import Constants

sac_config = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "buffer_size": int(1e6),
    "learning_starts": 10_000,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 256,
    "gradient_steps": 1,
    "policy_kwargs": {
        "activation_fn": torch.nn.Tanh,
        "net_arch": [256, 256]
    }
}


def get_run_name(algorithm: str, environment: str) -> str:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H_%M_%S")

    return f"{environment}_{algorithm}_{timestamp}"


def get_tensorboard_logs_directory(algorithm: str) -> str:
    return f"{Constants.LOGS_DIRECTORY}/{algorithm}/"


# Usefull link about evaluating models
# https://github.com/hill-a/stable-baselines/issues/376


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        update_value = self.update_value = self.model.n_envs
        self.progress_bar.update(update_value)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


class CustomEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(self, eval_env, n_eval_episodes=10,
                 eval_freq=10000, deterministic=True, verbose=0):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                 eval_freq=eval_freq,
                                                 deterministic=deterministic,
                                                 verbose=verbose)

        self._used_std = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.actor._last_used_log_std is not None:
                std = np.power(10, self.model.actor._last_used_log_std)
                self._used_std.append(std)
                self.logger.record('last_used_std', std)

        return super(CustomEvalCallback, self)._on_step()


if __name__ == '__main__':
    env_id = "Humanoid-v2"
    algorithm = "SAC"
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    eval_env = gym.make(env_id)

    tensorboard_log = get_tensorboard_logs_directory(algorithm)

    # dd/mm/YY
    eval_callback = CustomEvalCallback(eval_env, n_eval_episodes=10, eval_freq=1000, deterministic=True, verbose=True)

    print(type(eval_callback))
    model = SAC(env=env, tensorboard_log=tensorboard_log, verbose=False, device="cuda", **sac_config)
    model.learn(total_timesteps=int(5e6), callback=[eval_callback, TqdmCallback()],
                tb_log_name=get_run_name(algorithm=algorithm, environment=env_id))

    used_stds = np.power((np.ones_like(eval_callback._used_std) * 10), eval_callback._used_std)
    plt.plot(used_stds)
    plt.show()

    mean = np.mean(used_stds)
    median = np.median(used_stds)

    print(f"Mean of std used = {mean}")
    print(f"Median of std used = {median}")
