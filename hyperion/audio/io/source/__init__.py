from abc import abstractmethod
from typing import Generator


class AudioSource:
    def __init__(self):
        self._generator: Generator = ...

    def __call__(self, *args, **kwargs):
        self._generator = self._init_generator()
        return self._generator

    @abstractmethod
    def _init_generator(self) -> Generator:
        # yield something here
        pass

