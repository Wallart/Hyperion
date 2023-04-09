from time import time
from utils.threading import Consumer

import logging
import numpy as np
import sounddevice as sd


class SpeakersStream(Consumer):

    def __init__(self, sample_rate, channels):
        super().__init__()
        self.sample_rate = sample_rate
        # self._speakers = sd.OutputStream(sample_rate, channels=channels, dtype=np.float32, callback=)
        # callback(outdata: numpy.ndarray, frames: int, time: CData, status: CallbackFlags) -> None

    # def __del__(self):
    #     self._speakers.stop()
    #     self._speakers.close()

    def run(self) -> None:
        # self._speakers.start()
        while True:
            audio = self._in_queue.get()
            t0 = time()
            # self._speakers.write(audio)
            sd.play(audio, blocking=False, samplerate=self.sample_rate)
            logging.info(f'{self.__class__.__name__} {time() - t0} exec. time')
