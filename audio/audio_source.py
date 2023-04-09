from abc import ABC, abstractmethod


class AudioSource(ABC):

    def __init__(self, sampling_rate=16000, channels=1, duration_ms=512):
        self._generator = None

        self._channels = channels
        self._sampling_rate = sampling_rate
        self._chunk_duration_ms = duration_ms
        self._chunk_size = int(self._sampling_rate * self._chunk_duration_ms / 1000)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read(self):
        return self._generator

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def _init_generator(self):
        pass

    @abstractmethod
    def name(self):
        pass

    def chunk_size(self):
        return self._chunk_size

    def chunk_duration(self):
        return self._chunk_duration_ms

    def sample_rate(self):
        return self._sampling_rate

    def channels(self):
        return self._channels
