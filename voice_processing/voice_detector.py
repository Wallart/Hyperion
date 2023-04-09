from time import time
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import VAD

import os
import torch


class VoiceDetector(Consumer, Producer):

    def __init__(self, ctx, sampling_rate, model_path='~/.hyperion/vad', activation_threshold=.8):
        super().__init__()

        self._ctx = ctx
        self._sampling_rate = sampling_rate
        self._act_thresh = activation_threshold
        opts = {
            'source': 'speechbrain/vad-crdnn-libriparty',
            'savedir': os.path.expanduser(model_path),
            # 'run_opts': {'device': ctx[0]}
        }
        ProjectLogger().info(opts)
        self._vad = VAD.from_hparams(**opts).to(ctx[0])

        self._buffer = None

    def _detect(self, chunk):
        chunk = torch.tensor(chunk)#.to(self._ctx[0])
        prob = self._vad.get_speech_prob_chunk(chunk)
        prob_th = self._vad.apply_threshold(prob, activation_th=self._act_thresh)

        if prob_th.sum() > 0:
            boundaries = self._vad.get_boundaries(prob_th, output_value='frame')[0]
            start_index, end_index = [e.item() for e in boundaries]
            voice_chunk = chunk
            self._buffer = voice_chunk if self._buffer is None else torch.cat([self._buffer, voice_chunk])
            return True
        return False

    def _flush(self):
        if self._buffer is not None:
            sentence = self._buffer
            self._buffer = None
            self._dispatch(sentence)

    def run(self):
        while True:
            audio_chunk = self._in_queue.get()
            t0 = time()

            if audio_chunk is None:
                self._flush()
            else:
                self._detect(audio_chunk)

            ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
