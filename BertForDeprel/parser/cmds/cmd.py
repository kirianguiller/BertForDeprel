from abc import ABC, abstractmethod
from argparse import _SubParsersAction, ArgumentParser
from typing import Any, TYPE_CHECKING


from ..utils.types import ModelParams_T

# see https://github.com/python/typeshed/issues/7539
if TYPE_CHECKING:
    SubparsersType = _SubParsersAction[ArgumentParser]
else:
    SubparsersType = Any

class CMD(ABC):
    def __call__(self, args, model_params: ModelParams_T):
        # TODO: why are we saving these?
        self.args = args

    @abstractmethod
    def add_subparser(self, name: str, subparsers: SubparsersType) -> ArgumentParser:
        pass
