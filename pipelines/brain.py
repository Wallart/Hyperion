from time import time
from audio import int16_to_float32
from analysis.chat_gpt import ChatGPT
from utils.logger import ProjectLogger
from utils.protocol import frame_encode
from utils.request import RequestObject
from analysis.command_detector import CommandDetector, ACTIONS
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import queue
import numpy as np


class Brain:
    def __init__(self, ctx, opts, host='0.0.0.0'):
        self.host = host
        self.name = opts.name
        self.port = opts.port
        self.debug = opts.debug
        self.frozen = False
        self.sio = None

        # Raw audio analysis pipeline
        self.detector = VoiceDetector(ctx[-1:], 16000, activation_threshold=.9)
        self.recognizer = VoiceRecognizer(ctx[-1:])
        self.detector.pipe(self.recognizer)

        # logical thinking and speech synthesis block
        self.transcriber = VoiceTranscriber(ctx, opts.whisper)
        self.chat = ChatGPT(opts.name, opts.gpt, opts.no_memory, opts.clear, opts.prompt)
        self.synthesizer = VoiceSynthesizer()
        self.transcriber.pipe(self.chat).pipe(self.synthesizer)

        # commands handling block
        self.commands = CommandDetector()
        self.transcriber.pipe(self.commands)

        # intakes
        self.audio_intake = self.detector.create_intake()
        self.speech_intake = self.transcriber.create_intake()
        self.cmd_intake = self.commands.get_intake()
        self.chat_intake = self.chat.get_intake()

        # sinks
        self.audio_sink = self.recognizer.create_sink()
        self.cmd_sink = self.commands.create_sink()

        self.threads = [self.transcriber, self.commands, self.chat, self.synthesizer, self.recognizer, self.detector]

    def start(self, sio, flask_app):
        try:
            _ = [t.start() for t in self.threads]
            # flask_app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
            self.sio = sio
            self.sio.run(flask_app, host=self.host, debug=self.debug, port=self.port, allow_unsafe_werkzeug=True)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]
        self.sio.stop()

    def sink_streamer(self, sink):
        while True:
            if self.frozen:
                return

            # TODO Ship into drain ?
            try:
                request_obj = sink.drain()
            except queue.Empty:
                continue

            if request_obj.text_answer is None and request_obj.audio_answer is None and request_obj.termination:
                self.synthesizer.delete_identified_sink(request_obj.identifier)
                return

            if request_obj.audio_answer is None:
                # if text answer not empty but audio is, means speech synthesis api call failed
                ProjectLogger().warning('Speech synthesis failed.')
                request_obj.audio_answer = np.zeros((0,))

            yield frame_encode(request_obj.timestamp, request_obj.num_answer, request_obj.text_request, request_obj.text_answer, request_obj.audio_answer)

    def handle_speech(self, request_id, request_sid, speaker, speech):
        request_obj = RequestObject(request_id, speaker)
        request_obj.set_audio_request(speech)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.speech_intake.put(request_obj)

        self.handle_commands(request_id, request_sid, speaker, sink)

        stream = self.sink_streamer(sink)
        return stream

    def handle_chat(self, request_id, request_sid, user, message):
        request_obj = RequestObject(request_id, user)
        request_obj.set_text_request(message)

        sink = self.synthesizer.create_identified_sink(request_id)

        self.cmd_intake.put(request_obj)
        self.handle_commands(request_id, request_sid, user, sink)
        self.chat_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return stream

    def handle_audio(self, audio):
        buffer = np.frombuffer(audio, dtype=np.int16)
        buffer = int16_to_float32(buffer)

        self.audio_intake.put(buffer)
        self.audio_intake.put(None)  # end of speech

        # TODO Ship into drain ?
        speech, speaker = None, None
        while True:
            try:
                speech, speaker = self.audio_sink.drain()
                break
            except queue.Empty:
                continue

        speech = speech if speech is None else speech.numpy()
        return speaker, speech

    def handle_commands(self, request_id, request_sid, speaker, sink):
        # TODO Ship into drain ?
        detected_cmd = None
        while True:
            try:
                detected_cmd = self.cmd_sink.drain()
                break
            except queue.Empty:
                continue

        if detected_cmd == ACTIONS.SLEEP.value:
            self.frozen = True
            self.chat.frozen = True
        elif detected_cmd == ACTIONS.WAKE.value:
            self.frozen = False
            self.chat.frozen = False
            sink._sink.put(RequestObject(request_id, speaker, termination=True))  # to avoid deadlocked sink
        elif detected_cmd == ACTIONS.WIPE.value:
            self.chat.clear_context()
            ProjectLogger().warning('Memory wiped.')
        elif detected_cmd == ACTIONS.QUIET.value:
            sink._sink.put(RequestObject(request_id, speaker, termination=True, priority=0))
            self.sio.emit('interrupt', time(), to=request_sid)
