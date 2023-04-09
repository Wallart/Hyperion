from audio import int16_to_float32
from audio.io.source import AudioSource
from audio.io.pyaudio_resource import PyAudioResource

import audioop
import numpy as np
import noisereduce as nr

from utils.logger import ProjectLogger


class InDevice(PyAudioResource, AudioSource):

    def __init__(self, device_idx, sample_rate, duration=512):
        super().__init__(device_idx, False, sample_rate)
        AudioSource.__init__(self, sample_rate, duration)
        self._generator = self._init_generator()

    def _init_generator(self):
        listening = False
        listened_chunks = 0
        while self.opened:
            print(self._stream.is_stopped())
            raw_buffer = self._stream.read(self.chunk_size, exception_on_overflow=False)
            buffer = nr.reduce_noise(np.frombuffer(raw_buffer, dtype=np.int16), self.sample_rate)

            # root mean square of signal to detect if there is interesting things to record
            rms = audioop.rms(buffer, 2)
            if rms >= 1000:
                listening = True
            elif rms < 1000 and listened_chunks >= 4:  # eq to 2 sec of silence
                listening = False
                listened_chunks = 0
                ProjectLogger().info('Candidate noise detected.')

            if listening:
                listened_chunks += 1
                yield buffer
            else:
                yield -1  # sending silence token
        print('ok')