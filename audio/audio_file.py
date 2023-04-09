from time import sleep
from audio.audio_source import AudioSource

import os
import logging
import librosa
import numpy as np
# import soundfile as sd


class AudioFile(AudioSource):

    def __init__(self, wav_file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        wav_file = os.path.expanduser(wav_file)
        wav, _ = librosa.load(wav_file, sr=self._sampling_rate)
        self._wav = wav#.astype(np.int16)
        # self._wav, sample_rate = sd.read(wav_file, dtype=np.int16, samplerate=self._sampling_rate)
        self._file_name = os.path.basename(wav_file)

    def start(self):
        self._generator = self._init_generator()

    def close(self):
        pass

    def _init_generator(self):
        i = 0
        while True:
            end_index = min(i + self._chunk_size, len(self._wav))
            wav_chunk = self._wav[i:end_index]
            i = end_index
            i = i % len(self._wav)
            yield wav_chunk
            if i == 0:
                logging.info('wav file sent. Sleeping 30 secs')
                sleep(30)
                logging.info('sleep time is over.')

    def name(self):
        return self._file_name
