import queue
from time import time
from audio import int16_to_float32
from utils.logger import ProjectLogger
from utils.threading import Consumer
from audio.io.sound_device_resource import SoundDeviceResource

import logging


class AudioOutput(SoundDeviceResource, Consumer):

    def __init__(self, device_idx, sample_rate, **kwargs):
        super().__init__(device_idx, True, sample_rate, **kwargs)
        Consumer.__init__(self)
        self.sample_rate = sample_rate

    def stop(self):
        super().stop()
        self.close()

    def run(self) -> None:
        self.open()
        while self._stream.active:
            try:
                audio = self._in_queue.get(timeout=self._timeout)
                t0 = time()

                self._stream.write(int16_to_float32(audio))

                self._in_queue.task_done()
                logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
