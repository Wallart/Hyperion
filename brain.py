#!/usr/bin/env python

from time import time
from daemonocle import Daemon
from brain.chat_gpt import ChatGPT, CHAT_MODELS
from utils.logger import ProjectLogger
from utils.utils import get_ctx, frame_encode
from flask import Flask, Response, request, g
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import os
import argparse
import numpy as np


class Brain:
    def __init__(self, ctx, port, debug, name, model, memory, clear, prompt, host='0.0.0.0'):
        self.host = host
        self.port = port
        self.debug = debug

        transcriber = VoiceTranscriber(ctx)
        synthesizer = VoiceSynthesizer()
        chat = ChatGPT(name, model, memory, clear, prompt)

        self.intake_1, self.sink_1 = transcriber.create_intake(), transcriber.create_sink()
        self.intake_2, self.sink_2a, self.sink_2b = chat.create_intake(), chat.create_sink(), chat.pipe(synthesizer).create_sink()
        self.threads = [transcriber, synthesizer, chat]

    def boot(self):
        try:
            _ = [t.start() for t in self.threads]
            app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]

    @staticmethod
    def sink_streamer(request, text_sink, audio_sink):
        while True:
            text_chunk = text_sink.get()
            audio_chunk = audio_sink.get()
            if audio_chunk is None:
                return

            encoded_frame = frame_encode(request, text_chunk, audio_chunk)
            yield encoded_frame

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
        stream = Brain.sink_streamer(transcription, self.sink_2a, self.sink_2b)
        return Response(response=stream, mimetype='application/octet-stream')


APP_NAME = 'hyperion_brain'
app = Flask(__name__)


@app.route('/audio', methods=['POST'])
def audio_stream():
    return brain.handle_audio()


@app.route('/video')
def video_stream():
    return 'Not yet implemented', 500


@app.before_request
def before_request():
    g.start = time()


@app.after_request
def after_request(response):
    diff = time() - g.start
    ProjectLogger().info(f'Request execution time {diff:.3f} sec(s)')
    return response


def main():
    try:
        global brain
        ctx = get_ctx(args)
        brain = Brain(ctx, args.port, args.debug, args.name, args.gpt, args.no_memory, args.clear)
        brain.boot()
    except Exception as e:
        ProjectLogger().error(f'Fatal error occurred : {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s brain')
    parser.add_argument('-p', '--port', type=int, default=9999, help='Listening port.')
    parser.add_argument('--debug', action='store_true', help='Enables flask debugging.')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon.')
    parser.add_argument('--clear', action='store_true', help='Clean persistent memory at startup')
    parser.add_argument('--no-memory', action='store_true', help='Start bot without persistent memory.')
    parser.add_argument('--name', type=str, default='Hypérion', help='Set bot name.')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
    parser.add_argument('--gpt', type=str, default=CHAT_MODELS[1], choices=CHAT_MODELS, help='GPT version to use.')
    parser.add_argument('--prompt', type=str, default='base', help='Prompt file to use.')
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
