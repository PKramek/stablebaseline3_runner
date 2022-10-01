import logging
import warnings

import gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.Constants import Constants
from src.algorithm.algorithm_factory import AlgorithmFactory
from src.args_parser import StableBaselines3Parser
from src.args_transformer.args_transformer import PolicyArgsTransformer, ActivationFunctionArgsTransformer
from src.args_transformer.transformer_pipeline import ArgsTransformerPipeline
from src.results_creator.results_crator import ResultsCreator
from src.results_repository.csv_results_repository import CSVResultsRepository
from src.reward_shaping.entrypoint_factory import RewardShapingEntrypointCreatorFactory
from src.reward_shaping.fi.fi_factory import FiFactory
from src.utils import get_tensorboard_logs_directory, get_run_name, get_experiment_tb_directory

logging.basicConfig(format=Constants.LOG_FORMAT)
logger = logging.getLogger(Constants.LOGGER_NAME)
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')


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

    logger.info("Registering reward shaping environments...")
    for environment_name in Constants.ENVIRONMENTS_WITH_REWARD_SHAPING:
        for fi_name in FiFactory.FI_MAPPING:
            fi = FiFactory.get(name=fi_name)
            shaped_env_name = f"{fi_name}RewardShaping{environment_name}"

            entrypoint_creator = RewardShapingEntrypointCreatorFactory.get(name=environment_name)(
                shaped_env_name=shaped_env_name,
                gamma=algorithm_config["gamma"],
                fi=fi,
            )

            gym.envs.register(
                id=entrypoint_creator.shaped_env_name,
                entry_point=entrypoint_creator.create(),
            )
            logger.info(f"Registered {shaped_env_name} environment...")

    run_name = get_run_name(algorithm=general_config["algo"], environment=general_config["env"])
    tensorboard_log_directory = get_tensorboard_logs_directory(general_config["algo"])
    results_dir = get_experiment_tb_directory(tensorboard_logs_directory=tensorboard_log_directory, run_name=run_name)

    # StableBaselines3 adds _1 at the end of the directory
    extended_results_dir = f"{results_dir}_1"

    eval_env = gym.make(general_config["eval_env"])
    env = SubprocVecEnv([make_env(general_config["env"], i) for i in range(general_config["num_parallel_envs"])],
                        start_method='fork')

    logger.info(f"env name: {general_config['env']}")
    logger.info(f"eval_env name: {general_config['eval_env']}")

    eval_callback = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=1000, deterministic=True, verbose=True)

    model = AlgorithmFactory.get(general_config["algo"])(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log_directory,
        verbose=False,
        **algorithm_config)

    logger.info("Starting the learning process...")
    model.learn(total_timesteps=general_config["max_timesteps"],
                callback=[eval_callback],
                tb_log_name=run_name)
    logger.info("Learning process ended, saving results...")

    results_creator = ResultsCreator(evaluation_callback_object=eval_callback)
    results_repository = CSVResultsRepository(results_dir=extended_results_dir)

    results_repository.save(results_creator.results_dict)

    logger.info(f"saved evaluation results in {extended_results_dir}")
