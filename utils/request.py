import numpy as np


class RequestObject:

    def __init__(self, identifier, user, termination=False):
        self.termination = termination
        self.identifier = identifier
        self.user = user

        self.text_request = None
        self.audio_request = None
        self.request_lang = None

        self.num_answer = 0
        self.text_answer = None
        self.audio_answer = None

    def set_audio_request(self, audio_buffer):
        self.audio_request = np.frombuffer(audio_buffer, dtype=np.float32)

    def set_text_request(self, text):
        self.text_request = text
