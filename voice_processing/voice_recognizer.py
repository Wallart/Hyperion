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

        self.sample_rate = 16000  # Model is using 16kHZ samples
        self._recog_threshold = recog_threshold
        self._speakers_sample_dir = os.path.join(os.getcwd(), 'resources', 'speakers_samples')
        self._recog = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb')

    def load_wavfile(self, file_path):
        wav, _ = librosa.load(file_path, sr=self.sample_rate)
        return wav

    def recognize(self, audio_chunk):
        if type(audio_chunk) == np.ndarray:
            audio_chunk = torch.tensor(audio_chunk)

        list_speakers = os.listdir(self._speakers_sample_dir)
        for speaker in list_speakers:
            speaker_file_samples = glob(os.path.join(self._speakers_sample_dir, speaker, '*.wav'))
            speaker_references = [self.load_wavfile(f) for f in speaker_file_samples]
            smallest_ref = min([len(r) for r in speaker_references])
            speaker_references = [r[:smallest_ref, ...] for r in speaker_references]

            speaker_references = np.stack(speaker_references, axis=0)
            audio_chunk = audio_chunk.unsqueeze(0).repeat(len(speaker_references), 1)

            scores, prediction = self._recog.verify_batch(torch.tensor(speaker_references), audio_chunk)
            if (prediction.int().sum() > 0).item():
                return speaker[0].upper() + speaker[1:]

        return 'Unknown'

    def run(self):
        while True:
            audio_chunk = self._in_queue.get()
            t0 = time()
            recognized_speaker = self.recognize(audio_chunk)
            logging.info(f'{recognized_speaker}\'s speaking...')
            self._dispatch((audio_chunk, recognized_speaker))
            logging.debug(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')


if __name__ == '__main__':
    test1 = os.path.expanduser('~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    test2 = os.path.join(os.getcwd(), 'resources/speakers_samples/test.wav')
    recognizer = VoiceRecognizer()
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test1)))
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test2)))
