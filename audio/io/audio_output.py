from time import time, sleep
from librosa import resample
from audio import int16_to_float32
from utils.threading import Consumer
from utils.logger import ProjectLogger
from audio.io.sound_device_resource import SoundDeviceResource

import queue
import logging


class AudioOutput(SoundDeviceResource, Consumer):

    def __init__(self, device_idx, sample_rate, **kwargs):
        super().__init__(device_idx, True, sample_rate, **kwargs)
        Consumer.__init__(self)
        self.sample_rate = sample_rate
        # self._previously_played = None
        self._interrupted = False
        self._interrupt_stamp = 0

    def stop(self):
        super().stop()
        self.close()

    def change(self, device_idx):
        ProjectLogger().info('Changing Output device.')
        self.device_idx = device_idx
        self.close()
        self._init_stream()

    def mute(self, timestamp):
        ProjectLogger().info('Silence required !')
        self._stream.abort(ignore_errors=True)

        with self._in_queue.mutex:
            self._in_queue.queue.clear()

        # TODO not thread safe ?
        self._interrupted = True
        self._interrupt_stamp = timestamp

    def run(self) -> None:
        while self.running:
            self.open()
            while self._stream.active:
                try:
                    timestamp, audio = self._consume()
                    t0 = time()

                    if timestamp <= self._interrupt_stamp:
                        ProjectLogger().info('Ignored invalid sentences.')
                        continue

                    audio = resample(int16_to_float32(audio), orig_sr=self.sample_rate, target_sr=self.device_default_sr)
                    # self._previously_played = audio
                    try:
                        self._stream.write(audio)
                    except Exception:
                        ProjectLogger().warning('Output audio stream not available.')

                    logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
                except queue.Empty:
                    continue

            if self._interrupted:
                self._interrupted = False
                ProjectLogger().info('Audio output interrupted.')
            else:
                ProjectLogger().info('Audio output closed.')
                sleep(1)
