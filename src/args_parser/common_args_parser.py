import argparse
from typing import Union, List

from src.Constants import Constants


class GeneralConfigArgumentsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Args parser for general parameters')
        self.parser.add_argument('--algo', type=str, help='Algorithm to be used', required=True,
                                 choices=Constants.ALGORITHMS.keys())
        self.parser.add_argument('--env', type=str, help='OpenAI Gym environment name', default="Humanoid-v2")
        self.parser.add_argument('--eval_env', type=str, help='OpenAI Gym environment name', default="Humanoid-v2")
        self.parser.add_argument('--num_parallel_envs', type=int, help='Number of CPU cores to use', required=True)
        self.parser.add_argument('--max_timesteps', type=int, help='Maximum number of timesteps', required=True)

    def parse_known_args(self, args: List[str] = None, namespace: Union[None, argparse.Namespace] = None):
        return self.parser.parse_known_args(args=args, namespace=namespace)


class CommonArgumentsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Args parser for common parameters for all algorithms')
        self.parser.add_argument('--gamma', type=float, help='Discount factor', required=False, default=0.99)
        self.parser.add_argument('--learning_rate', type=float, help='Learning rate', required=False,
                                 default=0.0003)
        self.parser.add_argument('--batch_size', type=int, help='Minibatch size for each gradient update', default=256)
        self.parser.add_argument('--net_arch', nargs='+', type=int, help='Network architecture for actor and critic',
                                 required=False, default=[256, 256])
        self.parser.add_argument('--activation_fn', type=str, help='Activation function used in hidden layers',
                                 default="relu", choices=Constants.ACTIVATION_FUNCTIONS_MAPPING.keys())

    def parse_known_args(self, args: List[str] = None, namespace: Union[None, argparse.Namespace] = None):
        return self.parser.parse_known_args(args=args, namespace=namespace)
