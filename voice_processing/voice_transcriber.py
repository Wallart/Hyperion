from time import time
from utils.threading import Consumer, Producer
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR

import torch
import logging
import whisper
import numpy as np


class VoiceTranscriber(Consumer, Producer):

    def __init__(self, model_path=None):
        super().__init__()

        # small gère mieux le franglais que base
        self._asr = whisper.load_model('small')

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
        logging.info(f'Detected language: {max(probs, key=probs.get)}')

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        transcription = whisper.decode(self._asr, mel, options).text
        # VoiceTranscriber.display(transcription)

        # output = self._asr2.transcribe_batch(wav, wav_len)
        # transcription = VoiceTranscriber.sanitize(output[0][0])
        # VoiceTranscriber.display(transcription)
        return transcription

    def run(self):
        while True:
            voice_chunk = self._in_queue.get()
            logging.info(f'Transcription started.')
            t0 = time()
            result = self.transcribe(voice_chunk)
            self._dispatch(result)
            logging.info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
