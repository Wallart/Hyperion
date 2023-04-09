from time import time
from glob import glob
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import SpeakerRecognition

import os
import torch
import queue
import librosa
import numpy as np


class VoiceRecognizer(Consumer, Producer):

    def __init__(self, ctx, model_path='~/.hyperion', recog_threshold=0.25):
        super().__init__()

        self.sample_rate = 16000  # Model is using 16kHZ samples
        self._ctx = ctx
        self._recog_threshold = recog_threshold

        sample_dir = os.path.join(os.getcwd(), 'resources', 'speakers_samples')
        self.speakers_references = self.load_references(sample_dir)

        opts = {
            'source': 'speechbrain/spkrec-ecapa-voxceleb',
            'savedir': os.path.expanduser(os.path.join(model_path, 'recog')),
            # 'run_opts': {'device': ctx[0]}
        }
        self._recog = SpeakerRecognition.from_hparams(**opts)

    def load_references(self, sample_dir):
        speakers_references = {}
        for speaker in os.listdir(sample_dir):
            if not os.path.isdir(os.path.join(sample_dir, speaker)):
                continue

            wav_files = glob(os.path.join(sample_dir, speaker, '*.wav'))
            if len(wav_files) == 0:
                continue

            pcm = [self.load_wavfile(w) for w in wav_files]
            smallest_pcm = min([len(p) for p in pcm])
            pcm = [p[:smallest_pcm] for p in pcm]
            speakers_references[speaker] = np.stack(pcm, axis=0)
        return speakers_references

    def load_wavfile(self, file_path):
        # TODO Librosa fires a warning. ResourceWarning: unclosed file
        wav, _ = librosa.load(file_path, sr=self.sample_rate)
        trimmed_wav, _ = librosa.effects.trim(wav)
        return trimmed_wav

    def recognize(self, audio_chunk):
        if type(audio_chunk) == np.ndarray:
            audio_chunk = torch.tensor(audio_chunk)

        audio_chunk = audio_chunk.unsqueeze(0)

        computed_scores = {}
        for speaker, references in self.speakers_references.items():
            audio_chunk_tmp = audio_chunk.repeat(len(references), 1)
            t0 = time()
            scores, _ = self._recog.verify_batch(torch.tensor(references), audio_chunk_tmp)
            ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} RECOG INF exec. time')
            computed_scores[speaker] = round(scores.max().item(), 4)

        speaker_idx = np.argmax(list(computed_scores.values()))
        speaker = list(computed_scores.keys())[speaker_idx]
        best_score = computed_scores[speaker]

        ProjectLogger().info(f'Speakers scores : {computed_scores}')
        if best_score < self._recog_threshold:
            return 'Unknown'

        return speaker[0].upper() + speaker[1:]

    def run(self):
        while self.running:
            try:
                audio_chunk = self._in_queue.get(timeout=self._timeout)
                t0 = time()
                recognized_speaker = self.recognize(audio_chunk)
                ProjectLogger().info(f'{recognized_speaker}\'s speaking...')
                self._dispatch((audio_chunk, recognized_speaker))

                self._in_queue.task_done()
                ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Voice Recognizer stopped.')


if __name__ == '__main__':
    test1 = os.path.expanduser('~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    test2 = os.path.join(os.getcwd(), 'resources/speakers_samples/test.wav')
    recognizer = VoiceRecognizer()
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test1)))
    recognizer.recognize(torch.tensor(recognizer.load_wavfile(test2)))
