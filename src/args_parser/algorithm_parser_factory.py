from src.args_parser.algorithm_parsers import PPOParser, SACParser, AlgorithmParser


class AlgorithmParserFactory:
    _ALGORITHM_PARSER_MAPPING = {
        "SB_SAC": SACParser,
        "SB_PPO": PPOParser
    }

    @staticmethod
    def get(name: str) -> AlgorithmParser:
        parser_cls = AlgorithmParserFactory._ALGORITHM_PARSER_MAPPING.get(name, None)

        if parser_cls is None:
            raise ValueError(
                f"Undefined AlgorithmParser name: {name}, available types are: {AlgorithmParserFactory._ALGORITHM_PARSER_MAPPING.keys()}")

        return parser_cls()
