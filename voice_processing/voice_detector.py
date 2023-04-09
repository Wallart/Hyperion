from utils.threading import Consumer, Producer
from speechbrain.pretrained import VAD

import torch


class VoiceDetector(Consumer, Producer):

    def __init__(self, sampling_rate, model_path=None, activation_threshold=.8, max_silence_duration=1.5):
        super().__init__()

        self._sampling_rate = sampling_rate
        self._act_thresh = activation_threshold
        self._vad = VAD.from_hparams(source='speechbrain/vad-crdnn-libriparty', savedir=model_path)

        self._buffer = None
        self._silence_duration = 0
        self._max_silence_duration = max_silence_duration

    def _detect(self, chunk):
        chunk = torch.tensor(chunk)
        prob = self._vad.get_speech_prob_chunk(chunk)
        prob_th = self._vad.apply_threshold(prob, activation_th=self._act_thresh)

        if prob_th.sum() > 0:
            boundaries = self._vad.get_boundaries(prob_th, output_value='frame')[0]
            start_index, end_index = [e.item() for e in boundaries]
            # voice_chunk = chunk[start_index:end_index]
            voice_chunk = chunk
            self._buffer = voice_chunk if self._buffer is None else torch.cat([self._buffer, voice_chunk])
            return True
        return False

    def _flush(self):
        if self._silence_duration >= self._max_silence_duration:
            self._silence_duration = 0
            if self._buffer is not None:
                sentence = self._buffer
                self._buffer = None
                # print(sentence.shape)
                self._dispatch(sentence)

    def run(self):
        while True:
            audio_chunk = self._in_queue.get()
            # norm_chunk = normalizer(torch.tensor(chunk), source.sample_rate()) # very slow
            if not self._detect(audio_chunk):
                self._silence_duration += len(audio_chunk) / self._sampling_rate
                self._flush()
