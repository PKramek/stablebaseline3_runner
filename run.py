import logging

import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.Constants import Constants
from src.algorithm.algorithm_factory import AlgorithmFactory
from src.args_parser import StableBaselines3Parser
from src.args_transformer.args_transformer import PolicyArgsTransformer, ActivationFunctionArgsTransformer
from src.args_transformer.transformer_pipeline import ArgsTransformerPipeline
from src.utils import get_tensorboard_logs_directory, get_run_name

logging.basicConfig(format=Constants.LOG_FORMAT)
logger = logging.getLogger(Constants.LOGGER_NAME)

logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# sac_config = {
#     "policy": "MlpPolicy",
#     "learning_rate": 0.0003,
#     "buffer_size": int(1e6),
#     "learning_starts": 10_000,
#     "batch_size": 64,
#     "tau": 0.005,
#     "gamma": 0.99,
#     "train_freq": 256,
#     "gradient_steps": 1,
#     "policy_kwargs": {
#         "activation_fn": torch.nn.ReLU,
#         "net_arch": [256, 256]
#     }
# }

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


if __name__ == '__main__':
    general_config, algorithm_config = StableBaselines3Parser().parse_args()
    args_transformer_pipeline = ArgsTransformerPipeline([ActivationFunctionArgsTransformer, PolicyArgsTransformer])

    general_config, algorithm_config = vars(general_config), args_transformer_pipeline.transform(vars(algorithm_config))

    logger.info(f"Using general config: {general_config}")
    logger.info(f"Using algorithm config: {algorithm_config}")

    env = SubprocVecEnv([make_env(general_config["env"], i) for i in range(general_config["num_parallel_envs"])])
    eval_env = gym.make(general_config["env"])
    tensorboard_log_directory = get_tensorboard_logs_directory(general_config["algo"])
    eval_callback = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=1000, deterministic=True, verbose=True)

    run_name = get_run_name(algorithm=general_config["algo"], environment=general_config["env"])

    model = AlgorithmFactory.get(general_config["algo"])(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log_directory,
        verbose=False,
        **algorithm_config)

    model.learn(total_timesteps=general_config["max_timesteps"],
                callback=[eval_callback],
                tb_log_name=run_name)
