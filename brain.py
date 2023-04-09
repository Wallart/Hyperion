from brain.chat_gpt import ChatGPT
from flask import Flask, Response, request
from utils import TEXT_SEPARATOR, CHUNK_SEPARATOR
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import logging
import argparse
import numpy as np


class Brain:
    def __init__(self, port, debug, host='0.0.0.0'):
        self.host = host
        self.port = port
        self.debug = debug

        self.transcriber = VoiceTranscriber()
        self.synthesizer = VoiceSynthesizer()
        self.chat = ChatGPT()

        self.intake_1, self.sink_1 = self.transcriber.create_intake(), self.transcriber.create_sink()
        self.intake_2, self.sink_2a, self.sink_2b = self.chat.create_intake(), self.chat.create_sink(), self.chat.pipe(self.synthesizer).create_sink()

    def boot(self):
        self.transcriber.start()
        self.synthesizer.start()
        self.chat.start()
        app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)

    @staticmethod
    def sink_streamer(request, text_sink, audio_sink):
        i = 0
        while True:
            text_chunk = text_sink.get()
            audio_chunk = audio_sink.get()
            if audio_chunk is None:
                return

            req = bytes(request, 'utf-8')
            resp = bytes(text_chunk, 'utf-8')
            audio_bytes = audio_chunk.tobytes()
            response = resp + TEXT_SEPARATOR + audio_bytes + CHUNK_SEPARATOR
            response = req + TEXT_SEPARATOR + response if i == 0 else response
            i += 1
            yield response

    def handle_audio(self):
        speech = request.files['speech'].read()
        speaker = request.files['speaker'].read().decode('utf-8')

        audio_chunk = np.frombuffer(speech, dtype=np.float32)
        self.intake_1.put(audio_chunk)
        transcription = self.sink_1.get()
        if transcription is None:
            return Response(response='Speak louder motherfucker !', status=204, mimetype='text/plain')

        chat_input = f'{speaker} : {transcription}'
        logging.info(chat_input)

        self.intake_2.put(chat_input)
        return Response(response=Brain.sink_streamer(transcription, self.sink_2a, self.sink_2b), status=200, mimetype='application/octet-stream')


app = Flask(__name__)


@app.route('/audio', methods=['POST'])
def audio_stream():
    return brain.handle_audio()


@app.route('/video')
def video_stream():
    return 'Not yet implemented', 500


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Hyperion\'s brain')
    parser.add_argument('--port', type=int, default=9999, help='Listening port')
    parser.add_argument('--debug', action='store_true', help='Enables flask debugging')
    args = parser.parse_args()

    brain = Brain(args.port, args.debug)
    brain.boot()
