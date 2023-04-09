from time import time
from audio import int16_to_float32
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import VAD

import os
import torch
import queue


class VoiceDetector(Consumer, Producer):

    def __init__(self, ctx, sampling_rate, model_path='~/.hyperion', activation_threshold=.8):
        super().__init__()

        self._ctx = ctx
        self._sampling_rate = sampling_rate
        self._act_thresh = activation_threshold
        opts = {
            'source': 'speechbrain/vad-crdnn-libriparty',
            'savedir': os.path.expanduser(os.path.join(model_path, 'vad')),
            # 'run_opts': {'device': ctx[0]}
        }
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

        ProjectLogger().info(f'Noise rejected. {prob.mean().item() * 100:.2f}%')
        return False

    def _flush(self):
        if self._buffer is not None:
            ProjectLogger().info('Speech detected.')
            sentence = self._buffer
            self._buffer = None
            self._dispatch(sentence)

    def run(self):
        while self.running:
            try:
                task = self._in_queue.get(timeout=self._timeout)
                t0 = time()
                if task is None:  # silence token received
                    self._flush()
                elif self._detect(task):
                    ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} DETECT exec. time')

                self._in_queue.task_done()
            except queue.Empty:
                continue

        ProjectLogger().info('Voice Detector stopped.')
