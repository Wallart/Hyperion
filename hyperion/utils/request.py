from hyperion.utils.timer import Timer
from hyperion.audio import int16_to_float32

import numpy as np


class RequestObject:

    def __init__(self, identifier, user, termination=False, priority=1):
        self.priority = priority
        self.termination = termination
        self.identifier = identifier
        self.user = user

        self.timestamp = Timer().now()

        self.text_request = None
        self.audio_request = None
        self.request_lang = None
        self.preprompt = None
        self.llm = None

        self.voice = None
        self.speech_engine = None

        self.num_answer = 0
        self.text_answer = None
        self.audio_answer = None

    def set_audio_request(self, audio_buffer):
        if type(audio_buffer) == np.ndarray:
            assert audio_buffer.dtype == np.float32
            self.audio_request = audio_buffer
        else:
            self.audio_request = int16_to_float32(np.frombuffer(audio_buffer, dtype=np.int16))

    def set_speech_engine(self, engine):
        self.speech_engine = engine

    def set_voice(self, voice):
        self.voice = voice

    def set_text_request(self, text):
        self.text_request = text

    def set_preprompt(self, preprompt):
        if preprompt != '':
            self.preprompt = preprompt

    def set_llm(self, model):
        if model != '':
            self.llm = model

    def __eq__(self, other):
        return self.priority == other.priority

    def __gt__(self, other):
        return self.priority > other.priority
