from time import time
from utils.threading import Consumer
from audio.io.pyaudio_resource import PyAudioResource

import logging


class AudioOutput(PyAudioResource, Consumer):

    def __init__(self, device_idx, sample_rate, *args, **kwargs):
        super().__init__(device_idx, True, sample_rate, *args, **kwargs)
        Consumer.__init__(self)
        self.sample_rate = sample_rate

    def stop(self):
        super().stop()
        self.close()

    def run(self) -> None:
        self.open()
        while self.running:
            audio = self._in_queue.get()
            t0 = time()

            self._stream.write(audio, len(audio))
            logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
