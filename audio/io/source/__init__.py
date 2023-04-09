from abc import abstractmethod
from typing import Generator


class AudioSource:
    def __init__(self, sample_rate, duration_ms):
        self.duration_ms = duration_ms
        self.chunk_size = int(sample_rate * self.duration_ms / 1000)
        self._generator: Generator = ...

    def __call__(self, *args, **kwargs):
        return self._init_generator()

    @abstractmethod
    def _init_generator(self) -> Generator:
        # yield something here
        pass

