#!/usr/bin/python

from daemonocle import Daemon
from brain.chat_gpt import ChatGPT
from flask import Flask, Response, request
from utils import TEXT_SEPARATOR, CHUNK_SEPARATOR
from utils.logger import ProjectLogger
from utils.utils import get_ctx
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import os
import argparse
import numpy as np


class Brain:
    def __init__(self, ctx, port, debug, host='0.0.0.0'):
        self.host = host
        self.port = port
        self.debug = debug

        self.transcriber = VoiceTranscriber(ctx)
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

        chat_input = None if transcription is None else f'{speaker} : {transcription}'
        transcription = '' if transcription is None else transcription
        if chat_input is None:
            ProjectLogger().info(f'{speaker} : <UNKNOWN>')
        else:
            ProjectLogger().info(chat_input)

        self.intake_2.put(chat_input)
        return Response(response=Brain.sink_streamer(transcription, self.sink_2a, self.sink_2b), status=200, mimetype='application/octet-stream')


APP_NAME = 'hyperion_brain'
app = Flask(__name__)


@app.route('/audio', methods=['POST'])
def audio_stream():
    return brain.handle_audio()


@app.route('/video')
def video_stream():
    return 'Not yet implemented', 500


def main():
    try:
        global brain
        ctx = get_ctx(args)
        brain = Brain(ctx, args.port, args.debug)
        brain.boot()
    except Exception as e:
        ProjectLogger().error(f'Fatal error occurred : {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s brain')
    parser.add_argument('-p', '--port', type=int, default=9999, help='Listening port')
    parser.add_argument('--debug', action='store_true', help='Enables flask debugging')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
    args = parser.parse_args()

    _ = ProjectLogger(args, APP_NAME)

    if args.daemon:
        pid_file = os.path.join(os.path.sep, 'tmp', f'{APP_NAME.lower()}.pid')
        if os.path.isfile(pid_file):
            ProjectLogger().error('Daemon already running.')
            exit(1)

        daemon = Daemon(worker=main, pid_file=pid_file)
        daemon.do_action('start')
    else:
        main()
