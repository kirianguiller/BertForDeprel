from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import TYPE_CHECKING, Any

# see https://github.com/python/typeshed/issues/7539
if TYPE_CHECKING:
    SubparsersType = _SubParsersAction[ArgumentParser]
else:
    SubparsersType = Any


class CMD(ABC):
    def run(self, args: Namespace):
        # TODO: why are we saving these?
        self.args = args

    @abstractmethod
    def add_subparser(self, name: str, subparsers: SubparsersType) -> ArgumentParser:
        pass
