import os.path
import queue
from time import time
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR

import torch
import whisper
import numpy as np


class VoiceTranscriber(Consumer, Producer):

    def __init__(self, ctx, model_size='small', confidence_threshold=.7, model_path='~/.hyperion'):
        super().__init__()

        self._ctx = ctx
        self._confidence_threshold = confidence_threshold
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        assert model_size in valid_sizes

        # small gère mieux le franglais que base
        self._asr = whisper.load_model('medium', download_root=os.path.expanduser(os.path.join(model_path, 'whisper')), device=ctx[0])

        # self._asr2 = EncoderASR.from_hparams(source='speechbrain/asr-wav2vec2-commonvoice-fr', savedir=model_path)
        # self._asr3 = EncoderDecoderASR.from_hparams(source='speechbrain/asr-crdnn-commonvoice-fr', savedir=model_path)

    def transcribe(self, voice_chunk):
        if type(voice_chunk) == np.ndarray:
            voice_chunk = torch.tensor(voice_chunk)

        # pad/trim it to fit 30 seconds
        # I cannot speak without breathing more thant 12 seconds.
        audio = whisper.pad_or_trim(voice_chunk)
        # make log-mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self._asr.device)
        # detect the spoken language
        _, probs = self._asr.detect_language(mel)
        lang = max(probs, key=probs.get)
        score = probs[lang]
        ProjectLogger().info(f'Detected language -> {lang.upper()} {score * 100:.2f}%')

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        transcription = whisper.decode(self._asr, mel, options).text
        ProjectLogger().info(f'Transcription -> {transcription}')
        return transcription, lang, score

    def run(self):
        while self.running:
            try:
                request_obj = self._consume()

                ProjectLogger().info('Transcribing voice...')
                t0 = time()
                text, lang, score = self.transcribe(request_obj.audio_request)

                request_obj.text_request = text
                request_obj.request_lang = lang
                if score < self._confidence_threshold:
                    ProjectLogger().info(f'Score too low !')
                    request_obj.text_request = ''

                self._dispatch(request_obj)
                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Transcriber stopped.')
