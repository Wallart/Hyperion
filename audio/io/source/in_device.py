from audio import float32_to_int16
from utils.logger import ProjectLogger
from audio.io.source import AudioSource
from audio.io.sound_device_resource import SoundDeviceResource

import audioop
import noisereduce as nr


class InDevice(SoundDeviceResource, AudioSource):

    def __init__(self, device_idx, sample_rate, **kwargs):
        super().__init__(device_idx, False, sample_rate, **kwargs)

    def _init_generator(self):
        listening = False
        listened_chunks = 0
        while self._stream.active:
            buffer, overflowed = self._stream.read(self.chunk_size)
            buffer = buffer.squeeze()
            buffer = nr.reduce_noise(buffer, self.sample_rate)

            # root mean square of signal to detect if there is interesting things to record
            rms = audioop.rms(float32_to_int16(buffer), 2)
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
                yield None  # sending silence token