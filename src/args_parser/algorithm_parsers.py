import argparse
from abc import ABC, abstractmethod
from typing import List, Union


class AlgorithmParser(ABC):

    @abstractmethod
    def parse_args(self, args: List[str], namespace: Union[None, argparse.Namespace]):
        raise NotImplementedError()


class SACParser(AlgorithmParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Args parser for SAC algorithm')
        self.parser.add_argument('--learning_starts', type=int, help='Experience replay warm start coefficient',
                                 default=1000)
        self.parser.add_argument('--buffer_size', type=int, help='Memory buffer size', required=False, default=int(1e6))
        self.parser.add_argument('--tau', type=float, help='The soft update coefficient (Polyak update)',
                                 required=False, default=0.005)
        self.parser.add_argument('--train_freq', type=int, help='Update the model every <train_freq> steps.', default=1)
        self.parser.add_argument('--gradient_steps', type=int,
                                 help=' How many gradient steps to do after each rollout.', default=1)

    def parse_args(self, args: List[str] = None, namespace: Union[None, argparse.Namespace] = None):
        return self.parser.parse_args(args=args, namespace=namespace)


class PPOParser(AlgorithmParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Args parser for SAC algorithm')

        self.parser.add_argument('--n_steps', type=int,
                                 help='The number of steps to run for each environment per update.', default=2048)
        self.parser.add_argument('--n_epochs', type=int,
                                 help='Number of epoch when optimizing the surrogate loss.', default=10)
        self.parser.add_argument('--gae_lambda', type=float,
                                 help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator.',
                                 required=False, default=0.95)
        self.parser.add_argument('--clip_range', type=float,
                                 help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator.',
                                 required=False, default=0.2)

    def parse_args(self, args: List[str] = None, namespace: Union[None, argparse.Namespace] = None):
        return self.parser.parse_args(args=args, namespace=namespace)
