from time import time
from speechbrain.pretrained import VAD
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.threading import Consumer, Producer

import torch
import queue


class VoiceDetector(Consumer, Producer):

    def __init__(self, ctx, sampling_rate, activation_threshold=.8):
        super().__init__()

        self._ctx = ctx
        self._sampling_rate = sampling_rate
        self._act_thresh = activation_threshold
        opts = {
            'source': 'speechbrain/vad-crdnn-libriparty',
            'savedir': ProjectPaths().cache_dir / 'vad',
            # Model is so small and fast that CPU has been hardcoded in hyperparams.yaml from the savedir
            # 'run_opts': {'device': ctx[0]}
        }
        self._vad = VAD.from_hparams(**opts)

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
                task = self._consume()
                t0 = time()
                if task is None:  # silence token received
                    self._flush()
                elif self._detect(task):
                    ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} DETECT exec. time')
                else:
                    self._dispatch(None)  # To avoid locks

            except queue.Empty:
                continue

        ProjectLogger().info('Voice Detector stopped.')
