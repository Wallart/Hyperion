from time import time
from glob import glob

from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import SpeakerRecognition

import os
import torch
import librosa
import numpy as np


class VoiceRecognizer(Consumer, Producer):

    def __init__(self, model_path='~/.hyperion/recog', recog_threshold=0.25):
        super().__init__()

        self.sample_rate = 16000  # Model is using 16kHZ samples
        self._recog_threshold = recog_threshold
        self._speakers_sample_dir = os.path.join(os.getcwd(), 'resources', 'speakers_samples')
        self._recog = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=os.path.expanduser(model_path))

    def load_wavfile(self, file_path):
        wav, _ = librosa.load(file_path, sr=self.sample_rate)
        return wav

    def recognize(self, audio_chunk):
        if type(audio_chunk) == np.ndarray:
            audio_chunk = torch.tensor(audio_chunk)

        audio_chunk = audio_chunk.unsqueeze(0)
        list_speakers = [e for e in os.listdir(self._speakers_sample_dir) if os.path.isdir(os.path.join(self._speakers_sample_dir, e))]

        computed_scores = {}
        for speaker in list_speakers:
            speaker_file_samples = glob(os.path.join(self._speakers_sample_dir, speaker, '*.wav'))
            if len(speaker_file_samples) == 0:
                continue

            speaker_references = [self.load_wavfile(f) for f in speaker_file_samples]
            smallest_ref = min([len(r) for r in speaker_references])
            speaker_references = [r[:smallest_ref] for r in speaker_references]

            speaker_references = np.stack(speaker_references, axis=0)
            audio_chunk = audio_chunk.repeat(len(speaker_references), 1)

            scores, _ = self._recog.verify_batch(torch.tensor(speaker_references), audio_chunk)
            computed_scores[speaker] = round(scores.max().item(), 4)

        speaker_idx = np.argmax(list(computed_scores.values()))
        speaker = list_speakers[speaker_idx]
        best_score = computed_scores[speaker]

        ProjectLogger().info(f'Speakers scores : {computed_scores}')
        if best_score < self._recog_threshold:
            return 'Unknown'

        return speaker[0].upper() + speaker[1:]

    def run(self):
        while True:
            audio_chunk = self._in_queue.get()
            t0 = time()
            recognized_speaker = self.recognize(audio_chunk)
            ProjectLogger().info(f'{recognized_speaker}\'s speaking...')
            self._dispatch((audio_chunk, recognized_speaker))
            ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')


if __name__ == '__main__':
    test1 = os.path.expanduser('~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    test2 = os.path.join(os.getcwd(), 'resources/speakers_samples/test.wav')
    recognizer = VoiceRecognizer()
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test1)))
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test2)))
