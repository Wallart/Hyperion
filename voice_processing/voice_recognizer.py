from time import time
from glob import glob
from utils.threading import Consumer, Producer
from speechbrain.pretrained import SpeakerRecognition

import os
import torch
import librosa
import logging
import numpy as np


class VoiceRecognizer(Consumer, Producer):

    def __init__(self, recog_threshold=0.7):
        super().__init__()

        self.sample_rate = 16000
        self._recog_threshold = recog_threshold
        self._audio_files = glob(os.path.join(os.getcwd(), 'resources', 'speakers_samples', '*.wav'))
        self._recog = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb')

    def load_wavfile(self, file_path):
        wav, _ = librosa.load(file_path, sr=self.sample_rate)
        return wav

    def recognize(self, audio_chunk):
        if type(audio_chunk) == np.ndarray:
            audio_chunk = torch.tensor(audio_chunk)

        # MUST be resampled to 16000
        references = np.stack([self.load_wavfile(f) for f in self._audio_files], axis=0)
        audio_chunk = audio_chunk.unsqueeze(0).repeat(len(references), 1)

        scores, prediction = self._recog.verify_batch(torch.tensor(references), audio_chunk)
        return (prediction.int().sum() > 0).item()

    def run(self):
        while True:
            audio_chunk = self._in_queue.get()
            t0 = time()
            recognized = self.recognize(audio_chunk)
            logging.info(f'Speaker recognized : {recognized}')
            self._dispatch((audio_chunk, recognized))
            logging.debug(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')


if __name__ == '__main__':
    test1 = os.path.expanduser('~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    test2 = os.path.join(os.getcwd(), 'resources/speakers_samples/test.wav')
    recognizer = VoiceRecognizer()
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test1)))
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test2)))
