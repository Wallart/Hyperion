from time import sleep
from sys import stdout
from utils.threading import Consumer
from transformers import pipeline
from speechbrain.pretrained import EncoderDecoderASR

import torch
import numpy as np


class VoiceTranscriber(Consumer):

    def __init__(self, model_path=None):
        super().__init__()

        # model = 'speechbrain/asr-wav2vec2-commonvoice-fr'
        model = 'speechbrain/asr-crdnn-commonvoice-fr'
        self._asr = EncoderDecoderASR.from_hparams(source=model, savedir=model_path)

        with torch.no_grad():
            self._pipe = pipeline('automatic-speech-recognition', 'facebook/wav2vec2-large-xlsr-53-french')

    def transcribe(self, voice_chunk):
        if type(voice_chunk) == np.ndarray:
            voice_chunk = torch.tensor(voice_chunk)

        wav = voice_chunk.unsqueeze(0)#.unsqueeze(-1)
        wav_len = torch.ones((1,))
        output = self._asr.transcribe_batch(wav, wav_len)
        transcription = output[0][0]
        VoiceTranscriber.display(transcription)

        transcription = self._pipe(voice_chunk.numpy())['text']
        VoiceTranscriber.display(transcription)

    def run(self):
        while True:
            voice_chunk = self._in_queue.get()
            self.transcribe(voice_chunk)

    @staticmethod
    def display(transcription):
        if len(transcription) > 0:
            for i in range(len(transcription)):
                char = transcription[i].lower() if i > 0 else transcription[i].upper()

                in_sentence_with_letters_around = 0 < i < len(transcription) - 1 and transcription[i - 1] != '' and transcription[i + 1] != ''
                one_letter_word = i - 2 < 0 or transcription[i - 2] == ' '
                is_space = transcription[i] == ' '
                if is_space and in_sentence_with_letters_around and one_letter_word:
                    char = '\''
                stdout.write(char)
                stdout.flush()
                sleep(.05)
            print('.')
