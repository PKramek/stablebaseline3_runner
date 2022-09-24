from typing import List

from src.args_transformer.args_transformer import ArgsTransformerBase


class ArgsTransformerPipeline:

    def __init__(self, transformers: List[ArgsTransformerBase]):
        self.transformers = transformers

    @property
    def transformers(self):
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: List[ArgsTransformerBase]):
        if not all(issubclass(x, ArgsTransformerBase) for x in transformers):
            raise ValueError("All elements of the transformers list must implement ArgsTransformerBase interface")

        self._transformers = transformers

    def transform(self, args_dict: dict) -> dict:
        for transformer in self.transformers:
            args_dict = transformer.transform(args_dict)

        return args_dict
