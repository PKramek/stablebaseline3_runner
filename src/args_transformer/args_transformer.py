from abc import ABC, abstractmethod

from src.Constants import Constants


class ArgsTransformerBase(ABC):

    @staticmethod
    @abstractmethod
    def transform(args_dict: dict) -> dict:
        raise NotImplementedError()


class ActivationFunctionArgsTransformer(ArgsTransformerBase):

    @staticmethod
    def transform(args_dict: dict) -> dict:
        activation_fn = Constants.ACTIVATION_FUNCTIONS_MAPPING[args_dict["activation_fn"]]

        args_dict["activation_fn"] = activation_fn

        return args_dict


class PolicyArgsTransformer(ArgsTransformerBase):
    _POLICY_DICT_KEY = "policy_kwargs"
    _POLICY_ARGS = {"activation_fn", "net_arch"}

    @staticmethod
    def transform(args_dict: dict) -> dict:
        policy_args_dict = {key: value for key, value in args_dict.items() if key in PolicyArgsTransformer._POLICY_ARGS}
        non_policy_args = {key: value for key, value in args_dict.items() if
                           key not in PolicyArgsTransformer._POLICY_ARGS}

        output = non_policy_args
        output[PolicyArgsTransformer._POLICY_DICT_KEY] = policy_args_dict

        return output
