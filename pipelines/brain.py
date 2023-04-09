from flask_socketio import emit
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

import numpy as np


class Brain:
    def __init__(self, ctx, opts, host='0.0.0.0'):
        self.host = host
        self.name = opts.name
        self.port = opts.port
        self.debug = opts.debug
        self.frozen = False

        # speech detection block
        self.detector = VoiceDetector(ctx[-1:], 16000, activation_threshold=.9)
        self.recognizer = VoiceRecognizer(ctx[-1:])

        self.audio_intake = self.detector.create_intake()
        self.speech_sink = self.detector.pipe(self.recognizer).create_sink()

        self.transcriber = VoiceTranscriber(ctx, opts.whisper)
        self.chat = ChatGPT(opts.name, opts.gpt, opts.no_memory, opts.clear, opts.prompt)
        self.synthesizer = VoiceSynthesizer()

        self.transcriber.pipe(self.chat).pipe(self.synthesizer)

        # commands handling block
        self.commands = CommandDetector()
        self.transcriber.pipe(self.commands)
        self.cmd_sink = self.commands.create_sink()

        self.speech_intake = self.transcriber.create_intake()
        self.text_intake = self.chat.get_intake()

        self.threads = [self.transcriber, self.commands, self.chat, self.synthesizer, self.recognizer, self.detector]

    def start(self, socketio, flask_app):
        try:
            _ = [t.start() for t in self.threads]
            # app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
            socketio.run(flask_app, host=self.host, debug=self.debug, port=self.port)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]

    def sink_streamer(self, sink):
        while True:
            if self.frozen:
                return

            request_obj = sink.drain()

            if request_obj.text_answer is None and request_obj.audio_answer is None and request_obj.termination:
                self.synthesizer.delete_identified_sink(request_obj.identifier)
                return

            if request_obj.audio_answer is None:
                # if text answer not empty but audio is, means speech synthesis api call failed
                ProjectLogger().warning('Speech synthesis failed.')
                request_obj.audio_answer = np.zeros((0,))

            yield frame_encode(request_obj.num_answer, request_obj.text_request, request_obj.text_answer, request_obj.audio_answer)

    def handle_speech(self, request_id, speaker, speech):
        request_obj = RequestObject(request_id, speaker)
        request_obj.set_audio_request(speech)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.speech_intake.put(request_obj)

        detected_cmd = self.cmd_sink.drain()
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
            emit('interrupt', to=request_id)

        stream = self.sink_streamer(sink)
        return stream

    def handle_chat(self, request_id, user, message):
        request_obj = RequestObject(request_id, user)
        request_obj.set_text_request(message)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.text_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return stream

    def handle_audio(self, audio):
        buffer = np.frombuffer(audio, dtype=np.int16)
        buffer = int16_to_float32(buffer)

        self.audio_intake.put(buffer)
        self.audio_intake.put(None)  # end of speech

        speech, speaker = self.speech_sink.drain()
        return speaker, speech.numpy()
