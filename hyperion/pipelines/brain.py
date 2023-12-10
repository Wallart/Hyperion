from PIL import Image
from pypdf import PdfReader
from hyperion.utils.paths import ProjectPaths
from hyperion.audio import int16_to_float32
from hyperion.analysis.chat_gpt import ChatGPT
from hyperion.utils.protocol import frame_encode
from hyperion.utils.request import RequestObject
from hyperion.video.image_generator import ImageGenerator
from hyperion.voice_processing.voice_detector import VoiceDetector
from hyperion.voice_processing.voice_recognizer import VoiceRecognizer
from hyperion.analysis.user_command_detector import UserCommandDetector
from hyperion.voice_processing.voice_synthesizer import VoiceSynthesizer
from hyperion.voice_processing.voice_transcriber import VoiceTranscriber
from hyperion.video.visual_question_answering import VisualQuestionAnswering
from hyperion.analysis.interpreted_command_detector import InterpretedCommandDetector

import io
import queue
import numpy as np


class Brain:
    def __init__(self, ctx, opts, host='0.0.0.0'):
        self.host = host
        self.name = opts.name
        self.port = opts.port
        self.debug = opts.debug
        self.sio = None

        # Raw audio analysis pipeline
        self.voice_detector = VoiceDetector(ctx[-1:], 16000, activation_threshold=.9)
        self.voice_recognizer = VoiceRecognizer(ctx[-1:])

        # logical thinking and speech synthesis block
        self.voice_transcriber = VoiceTranscriber(ctx, opts.whisper)
        self.chat_gpt = ChatGPT(opts.name, opts.gpt, opts.no_memory, opts.clear, opts.prompt)
        self.voice_synthesizer = VoiceSynthesizer()

        # video/image processing
        self.visual_answering = VisualQuestionAnswering(ctx)
        self.images_gen = ImageGenerator(ctx)

        # commands handling block
        self.user_commands = UserCommandDetector(self.chat_gpt.clear_context, lambda: self.sio)
        self.interp_commands = InterpretedCommandDetector()

        # pipelines
        self.voice_detector.pipe(self.voice_recognizer)
        self.voice_transcriber.pipe(self.user_commands).pipe(self.chat_gpt).pipe(self.interp_commands).pipe(self.voice_synthesizer)

        # intakes
        self.voice_det_intake = self.voice_detector.create_intake()
        self.vqa_intake = self.visual_answering.create_intake(maxsize=1)
        self.images_gen_intake = self.images_gen.create_intake()
        self.voice_transcriber_intake = self.voice_transcriber.create_intake()
        self.user_cmd_intake = self.user_commands.get_intake()
        self.synthesizer_intake = self.voice_synthesizer.get_intake()

        # sinks
        self.voice_recognizer_sink = self.voice_recognizer.create_sink()
        self.vqa_sink = self.visual_answering.create_sink()

        # delegates
        self.visual_answering.set_chat_delegate(self.chat_gpt)
        self.interp_commands.set_chat_delegate(self.chat_gpt)
        self.user_commands.set_img_intake(self.images_gen_intake)
        self.interp_commands.set_img_delegate(self.images_gen)
        self.images_gen.set_synthesizer_intake(self.synthesizer_intake)

        self.threads = [
            self.voice_detector,
            self.voice_recognizer,
            self.voice_transcriber,
            self.voice_synthesizer,
            self.user_commands,
            self.interp_commands,
            self.chat_gpt,
            self.visual_answering,
            self.images_gen
        ]

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
        except KeyboardInterrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]
        self.sio.stop()

    def create_identified_sink(self, request_id):
        sink = self.voice_synthesizer.create_identified_sink(request_id)
        self.user_commands.set_identified_sink(request_id, sink)
        self.images_gen.set_identified_sink(request_id, sink)
        return sink

    def delete_identified_sink(self, request_id):
        self.voice_synthesizer.delete_identified_sink(request_id)
        self.user_commands.delete_identified_sink(request_id)
        self.images_gen.delete_identified_sink(request_id)

    def sink_streamer(self, sink):
        while True:
            if self.user_commands.frozen:
                return

            # TODO Ship into drain ?
            try:
                request_obj = sink.drain()
            except queue.Empty:
                continue

            if request_obj.termination:
                self.delete_identified_sink(request_obj.identifier)
                return

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

    @staticmethod
    def _customize_request(request_obj, preprompt, llm, speech_engine, voice, silent):
        request_obj.set_preprompt(preprompt)
        request_obj.set_llm(llm)
        request_obj.set_speech_engine(speech_engine)
        request_obj.set_voice(voice)
        request_obj.set_silent(silent)

    def handle_speech(self, request_id, request_sid, speaker, speech, preprompt=None, llm=None, speech_engine=None, voice=None, silent=False, indexes=[]):
        request_obj = RequestObject(request_id, speaker)
        request_obj.socket_id = request_sid
        request_obj.set_audio_request(speech)
        request_obj.set_indexes(indexes)
        Brain._customize_request(request_obj, preprompt, llm, speech_engine, voice, silent)

        sink = self.create_identified_sink(request_id)
        self.voice_transcriber_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return stream

    def handle_chat(self, request_id, request_sid, user, message, preprompt=None, llm=None, speech_engine=None, voice=None, silent=False, indexes=[]):
        request_obj = RequestObject(request_id, user)
        request_obj.socket_id = request_sid
        request_obj.set_text_request(message)
        request_obj.set_indexes(indexes)
        Brain._customize_request(request_obj, preprompt, llm, speech_engine, voice, silent)

        sink = self.create_identified_sink(request_id)
        self.user_cmd_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return stream

    def handle_audio(self, audio):
        buffer = np.frombuffer(audio, dtype=np.int16)
        buffer = int16_to_float32(buffer)

        self.voice_det_intake.put(buffer)
        self.voice_det_intake.put(None)  # end of speech

        # TODO Ship into drain ?
        speech, speaker = None, None
        while True:
            try:
                speech, speaker = self.voice_recognizer_sink.drain()
                break
            except queue.Empty:
                continue

        speech = speech if speech is None else speech.numpy()
        return speaker, speech

    def handle_frame(self, frame):
        jpg_image = Image.open(io.BytesIO(frame))
        self.vqa_intake.put(np.asarray(jpg_image))
        caption = ''
        while True:
            try:
                caption = self.vqa_sink.drain()
                break
            except queue.Empty:
                continue
        return caption

    def handle_document(self, binary_stream, preprompt=None):
        reader = PdfReader(binary_stream)
        pages = [page.extract_text() for page in reader.pages]
        self.chat_gpt.add_document_context(pages, preprompt)
