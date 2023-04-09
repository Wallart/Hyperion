from time import sleep
from utils.logger import ProjectLogger
from audio.io.source import AudioSource

import os
import librosa


class InFile(AudioSource):

    def __init__(self, wav_file, sample_rate, duration_ms=512):
        super().__init__(sample_rate, duration_ms)
        self.opened = False
        self.sample_rate = sample_rate

        self._chunk_size = 512
        wav_file = os.path.expanduser(wav_file)
        self._filename = os.path.basename(wav_file)
        self._wav, _ = librosa.load(wav_file, sr=self.sample_rate)

    def close(self):
        self.opened = False

    def run(self):
        self.opened = True

        i = 0
        while self.opened:
            end_index = min(i + self._chunk_size, len(self._wav))
            wav_chunk = self._wav[i:end_index]
            i = end_index
            i = i % len(self._wav)
            yield wav_chunk
            if i == 0:
                ProjectLogger().info('Wav file sent. Sleeping 30 secs')
                sleep(30)
                ProjectLogger().info('sleep time is over.')
