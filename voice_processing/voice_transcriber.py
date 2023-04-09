from time import sleep
from sys import stdout
from utils.threading import Consumer, Producer
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR

import torch
import numpy as np


class VoiceTranscriber(Consumer, Producer):

    def __init__(self, model_path=None):
        super().__init__()

        self._asr = EncoderASR.from_hparams(source='speechbrain/asr-wav2vec2-commonvoice-fr', savedir=model_path)
        # self._asr2 = EncoderDecoderASR.from_hparams(source='speechbrain/asr-crdnn-commonvoice-fr', savedir=model_path)

    def transcribe(self, voice_chunk):
        if type(voice_chunk) == np.ndarray:
            voice_chunk = torch.tensor(voice_chunk)

        wav = voice_chunk.unsqueeze(0)#.unsqueeze(-1)
        wav_len = torch.ones((1,))
        output = self._asr.transcribe_batch(wav, wav_len)
        transcription = VoiceTranscriber.sanitize(output[0][0])
        VoiceTranscriber.display(transcription)

        # output = self._asr2.transcribe_batch(wav, wav_len)
        # transcription = VoiceTranscriber.sanitize(output[0][0])
        # VoiceTranscriber.display(transcription)
        return transcription

    def run(self):
        while True:
            voice_chunk = self._in_queue.get()
            result = self.transcribe(voice_chunk)
            self._dispatch(result)

    @staticmethod
    def sanitize(transcription):
        transcription = list(transcription)
        for i in range(len(transcription)):
            transcription[i] = transcription[i].lower() if i > 0 else transcription[i].upper()
            in_sentence_with_letters_around = 0 < i < len(transcription) - 1 and transcription[i - 1] != '' and transcription[i + 1] != ''
            one_letter_word = i - 2 < 0 or transcription[i - 2] == ' '
            is_space = transcription[i] == ' '
            valid_letter = transcription[i - 1].lower() in 'jmstcndl'
            if is_space and in_sentence_with_letters_around and one_letter_word and valid_letter:
                transcription[i] = '\''
        return ''.join(transcription)

    @staticmethod
    def display(transcription):
        if len(transcription) > 0:
            for char in transcription:
                stdout.write(char)
                stdout.flush()
                sleep(.05)
            print('.')
