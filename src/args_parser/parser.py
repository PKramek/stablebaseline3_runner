import argparse
from typing import Union, List, Tuple

from src.args_parser.algorithm_parser_factory import AlgorithmParserFactory
from src.args_parser.common_args_parser import CommonArgumentsParser, GeneralConfigArgumentsParser


class StableBaselines3Parser:
    class C(argparse.Namespace):
        pass

    def parse_args(self, args: Union[None, List[str]] = None,
                   algorithm_config_namespace: Union[None, argparse.Namespace] = None,
                   general_config_namespace: Union[None, argparse.Namespace] = None) -> Tuple[
        argparse.Namespace, argparse.Namespace]:

        if algorithm_config_namespace is None:
            algorithm_config_namespace = self.C()

        if general_config_namespace is None:
            general_config_namespace = self.C()

        _, algorithm_config_args = GeneralConfigArgumentsParser().parse_known_args(
            args=args,
            namespace=general_config_namespace)

        _, algorithm_specific_args = CommonArgumentsParser().parse_known_args(
            args=algorithm_config_args,
            namespace=algorithm_config_namespace)

        algorithm_parser = AlgorithmParserFactory.get(general_config_namespace.algo)
        algorithm_parser.parse_args(args=algorithm_specific_args, namespace=algorithm_config_namespace)

        return general_config_namespace, algorithm_config_namespace
