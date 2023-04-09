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

        smallest_speaker = min([speaker_batch.shape[1] for speaker_batch in list(self.speakers_references.values())])
        speakers_batch = [speaker_batch[:, :smallest_speaker] for speaker_batch in list(self.speakers_references.values())]
        self.speakers_batch = np.concatenate(speakers_batch, axis=0)

        self.speakers_batch_indexes = {}
        prev_pos = 0
        for i, (speaker, batch) in enumerate(self.speakers_references.items()):
            if i > 0:
                prev_pos += len(list(self.speakers_references.values())[i - 1])
            self.speakers_batch_indexes[speaker] = (prev_pos, prev_pos + len(batch) - 1)

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

        audio_chunk = audio_chunk.unsqueeze(0).repeat(len(self.speakers_batch), 1)
        scores, _ = self._recog.verify_batch(torch.tensor(self.speakers_batch), audio_chunk)

        speakers_scores = {}
        for k, (start, end) in self.speakers_batch_indexes.items():
            speaker_scores = scores[start:end + 1, :]
            speaker_score = round(speaker_scores.max().item(), 4)
            speakers_scores[k] = speaker_score

        best_speaker_idx = np.argmax(list(speakers_scores.values()))
        best_score = list(speakers_scores.values())[best_speaker_idx]
        best_speaker = list(speakers_scores.keys())[best_speaker_idx]

        ProjectLogger().info(f'Speakers scores : {speakers_scores}')
        if best_score < self._recog_threshold:
            return 'Unknown'

        return best_speaker[0].upper() + best_speaker[1:]

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
