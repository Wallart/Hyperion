from time import time
from pyaudio import PyAudio, paInt16
from utils.threading import Consumer

import logging


class SpeakersStream(Consumer):

    def __init__(self, sample_rate, channels):
        super().__init__()
        self.sample_rate = sample_rate
        self._pyaudio = PyAudio()

        opts = {
            'input': False,
            'output': True,
            'start': False,
            'format': paInt16,
            'channels': channels,
            'rate': self.sample_rate
        }
        self._stream = self._pyaudio.open(**opts)

    def run(self) -> None:
        self._stream.start_stream()
        while True:
            audio = self._in_queue.get()
            t0 = time()

            self._stream.write(audio, len(audio))
            logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
