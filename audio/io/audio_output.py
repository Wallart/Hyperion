from time import time
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

    def stop(self):
        super().stop()
        self.close()

    def mute(self):
        ProjectLogger().info('Silence required !')
        self._stream.abort(ignore_errors=True)

        with self._in_queue.mutex:
            self._in_queue.queue.clear()

        # TODO not thread safe ?
        self._interrupted = True

    def run(self) -> None:
        self.open()
        while self._stream.active:
            try:
                idx, audio = self._consume()
                t0 = time()
                # TODO Checking sentence idx won't be enough. We should have uuid to requests
                if self._interrupted and idx == 0:
                    self._interrupted = False
                elif self._interrupted and idx > 0:
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
            ProjectLogger().info('Restarting output stream.')
            self.run()
