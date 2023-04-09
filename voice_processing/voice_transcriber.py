import os.path
from time import time
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR

import torch
import whisper
import numpy as np


class VoiceTranscriber(Consumer, Producer):

    def __init__(self, ctx, model_size='small', confidence_threshold=.7, model_path='~/.hyperion/whisper'):
        super().__init__()

        self._ctx = ctx
        self._confidence_threshold = confidence_threshold
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        assert model_size in valid_sizes

        # small gère mieux le franglais que base
        self._asr = whisper.load_model('medium', download_root=os.path.expanduser(model_path), device=ctx[0])

        # self._asr2 = EncoderASR.from_hparams(source='speechbrain/asr-wav2vec2-commonvoice-fr', savedir=model_path)
        # self._asr3 = EncoderDecoderASR.from_hparams(source='speechbrain/asr-crdnn-commonvoice-fr', savedir=model_path)

    def transcribe(self, voice_chunk):
        if type(voice_chunk) == np.ndarray:
            voice_chunk = torch.tensor(voice_chunk)

        # Better performances but less flexible than whisper (franglais)
        # wav = voice_chunk.unsqueeze(0)#.unsqueeze(-1)
        # wav_len = torch.ones((1,))
        # output = self._asr2.transcribe_batch(wav, wav_len)
        # transcription = VoiceTranscriber.sanitize(output[0][0])
        # VoiceTranscriber.display(transcription)

        # pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(voice_chunk)
        # make log-mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self._asr.device)
        # detect the spoken language
        _, probs = self._asr.detect_language(mel)
        lang = max(probs, key=probs.get)
        score = probs[lang]
        ProjectLogger().info(f'Detected language: {lang} {round(score, 4)}')

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        transcription = whisper.decode(self._asr, mel, options).text
        # VoiceTranscriber.display(transcription)

        # output = self._asr2.transcribe_batch(wav, wav_len)
        # transcription = VoiceTranscriber.sanitize(output[0][0])
        # VoiceTranscriber.display(transcription)
        return transcription, lang, score

    def run(self):
        while True:
            voice_chunk = self._in_queue.get()
            ProjectLogger().info('Transcribing voice...')
            t0 = time()
            text, lang, score = self.transcribe(voice_chunk)

            if score < self._confidence_threshold:
                ProjectLogger().info(f'Score ({score}) too low for : {text}')
                self._dispatch(None)
            else:
                self._dispatch(text)
            ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
