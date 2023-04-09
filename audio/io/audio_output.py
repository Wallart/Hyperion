from time import time
from audio import int16_to_float32
from utils.logger import ProjectLogger
from utils.threading import Consumer
from audio.io.sound_device_resource import SoundDeviceResource
from librosa import resample

import queue
import logging


class AudioOutput(SoundDeviceResource, Consumer):

    def __init__(self, device_idx, sample_rate, **kwargs):
        super().__init__(device_idx, True, sample_rate, **kwargs)
        Consumer.__init__(self)
        self.sample_rate = sample_rate
        # self._previously_played = None

    def stop(self):
        super().stop()
        self.close()

    def run(self) -> None:
        self.open()
        while self._stream.active:
            try:
                audio = self._consume()
                t0 = time()

                # audio = int16_to_float32(audio)
                audio = resample(int16_to_float32(audio), orig_sr=self.sample_rate, target_sr=self.device_default_sr)
                self._stream.write(audio)
                # self._previously_played = audio

                logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
