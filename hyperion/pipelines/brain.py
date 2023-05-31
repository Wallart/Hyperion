from PIL import Image
from hyperion.utils import ProjectPaths
from hyperion.audio import int16_to_float32
from hyperion.analysis.chat_gpt import ChatGPT
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.protocol import frame_encode
from hyperion.utils.request import RequestObject
from hyperion.video.image_generator import ImageGenerator
from hyperion.analysis.command_detector import CommandDetector
from hyperion.voice_processing.voice_detector import VoiceDetector
from hyperion.voice_processing.voice_recognizer import VoiceRecognizer
from hyperion.voice_processing.voice_synthesizer import VoiceSynthesizer
from hyperion.voice_processing.voice_transcriber import VoiceTranscriber
from hyperion.video.visual_question_answering import VisualQuestionAnswering

import io
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

        # video/image processing
        self.vqa = VisualQuestionAnswering(ctx, self.chat)
        self.images = ImageGenerator(ctx)

        # intakes
        self.audio_intake = self.detector.create_intake()
        self.video_intake = self.vqa.create_intake(maxsize=1)
        self.img_intake = self.images.create_intake()
        self.speech_intake = self.transcriber.create_intake()

        # commands handling block
        self.commands = CommandDetector(self.chat.clear_context, lambda: self.sio, self.img_intake)

        self.transcriber.pipe(self.commands).pipe(self.chat).pipe(self.synthesizer)
        self.cmd_intake = self.commands.get_intake()

        # sinks
        self.audio_sink = self.recognizer.create_sink()

        self.threads = [self.transcriber, self.commands, self.chat, self.synthesizer, self.recognizer, self.detector, self.vqa, self.images]

    def start(self, sio, flask_app):
        try:
            _ = [t.start() for t in self.threads]
            # flask_app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
            self.sio = sio
            ssl_path = ProjectPaths().resources_dir / 'ssl'
            ssl_context = (ssl_path / 'cert.pem', ssl_path / 'key.pem')
            opts = dict(
                host=self.host,
                debug=self.debug,
                port=self.port,
                allow_unsafe_werkzeug=True,
                ssl_context=ssl_context
            )
            self.sio.run(flask_app, **opts)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]
        self.sio.stop()

    def create_identified_sink(self, request_id):
        sink = self.synthesizer.create_identified_sink(request_id)
        self.commands.set_identified_sink(request_id, sink)
        self.images.set_identified_sink(request_id, sink)
        return sink

    def delete_identified_sink(self, request_id):
        self.synthesizer.delete_identified_sink(request_id)
        self.commands.delete_identified_sink(request_id)
        self.images.delete_identified_sink(request_id)

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
                self.delete_identified_sink(request_obj.identifier)
                return

            if request_obj.audio_answer is None:
                # if text answer not empty but audio is, means speech synthesis api call failed
                ProjectLogger().warning('Speech synthesis failed.')
                request_obj.audio_answer = np.zeros((0,))

            args = [
                request_obj.timestamp,
                request_obj.num_answer,
                request_obj.user,
                request_obj.text_request,
                request_obj.text_answer,
                request_obj.audio_answer,
                request_obj.image_answer
            ]
            yield frame_encode(*args)

    def _customize_request(self, request_obj, preprompt, llm, speech_engine, voice, silent):
        request_obj.set_preprompt(preprompt)
        request_obj.set_llm(llm)
        request_obj.set_speech_engine(speech_engine)
        request_obj.set_voice(voice)
        request_obj.set_silent(silent)

    def handle_speech(self, request_id, request_sid, speaker, speech, preprompt=None, llm=None, speech_engine=None, voice=None, silent=False):
        request_obj = RequestObject(request_id, request_sid, speaker)
        request_obj.socket_id = request_sid
        request_obj.set_audio_request(speech)
        self._customize_request(request_obj, preprompt, llm, speech_engine, voice, silent)

        sink = self.create_identified_sink(request_id)
        self.speech_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return stream

    def handle_chat(self, request_id, request_sid, user, message, preprompt=None, llm=None, speech_engine=None, voice=None, silent=False):
        request_obj = RequestObject(request_id, request_sid, user)
        request_obj.socket_id = request_sid
        request_obj.set_text_request(message)
        self._customize_request(request_obj, preprompt, llm, speech_engine, voice, silent)

        sink = self.create_identified_sink(request_id)
        self.cmd_intake.put(request_obj)

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

    def handle_frame(self, frame):
        jpg_image = Image.open(io.BytesIO(frame))
        frame = np.asarray(jpg_image)
        self.video_intake.put(frame)
