#!/usr/bin/env python

from time import time
from daemonocle import Daemon
from utils.logger import ProjectLogger
from utils.utils import get_ctx, frame_encode
from brain.chat_gpt import ChatGPT, CHAT_MODELS
from flask_log_request_id import RequestID, current_request_id
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber
from flask import Flask, Response, request, g, stream_with_context

import os
import argparse
import numpy as np


class Brain:
    def __init__(self, ctx, port, debug, name, model, memory, clear, prompt, host='0.0.0.0'):
        self.host = host
        self.port = port
        self.debug = debug

        transcriber = VoiceTranscriber(ctx)
        self.synthesizer = VoiceSynthesizer()
        self.chat = ChatGPT(name, model, memory, clear, prompt)

        self.intake_1, self.sink_1 = transcriber.create_intake(), transcriber.create_sink()
        self.intake_2, self.sink_2a, self.sink_2b = self.chat.create_intake(), self.chat.create_sink(), self.chat.pipe(self.synthesizer).create_sink()
        self.threads = [transcriber, self.synthesizer, self.chat]

    def boot(self):
        try:
            _ = [t.start() for t in self.threads]
            app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]

    @staticmethod
    def sink_streamer(user_request, text_sink, audio_sink):
        while True:
            text_answer_id, audio_answer_id = None, None
            text_answer, audio_answer = None, None

            while audio_answer_id != current_request_id():
                text_answer, text_answer_id = text_sink.get()
                audio_answer, audio_answer_id, creation_date = audio_sink.get()
                if audio_answer_id == current_request_id():
                    break

                ProjectLogger().warning(f'Expired answer -> {text_answer}')

            if text_answer is None:
                # ChatGPT sent a stop token.
                return

            if audio_answer is None:
                # if text answer not empty but audio is, means speech synthesis api call failed
                ProjectLogger().warning('Speech synthesis failed.')
                return

            # seems impossible...
            if text_answer_id != audio_answer_id:
                ProjectLogger().error('Text request_id differs from Audio request_id.')
                return

            encoded_frame = frame_encode(user_request, text_answer, audio_answer)
            yield encoded_frame

    def handle_audio(self):
        speech = request.files['speech'].read()
        speaker = request.files['speaker'].read().decode('utf-8')

        audio_chunk = np.frombuffer(speech, dtype=np.float32)
        self.intake_1.put((audio_chunk, current_request_id()))
        transcription, request_id = self.sink_1.get()

        chat_input = None if transcription is None else f'{speaker} : {transcription}'
        transcription = '' if transcription is None else transcription
        if chat_input is None:
            ProjectLogger().info(f'{speaker} : <UNKNOWN>')
        else:
            ProjectLogger().info(chat_input)

        self.intake_2.put((chat_input, request_id))
        stream = Brain.sink_streamer(transcription, self.sink_2a, self.sink_2b)
        return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


APP_NAME = 'hyperion_brain'
app = Flask(__name__)
RequestID(app)


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
        brain = Brain(ctx, args.port, args.debug, args.name, args.gpt, args.no_memory, args.clear, args.prompt)
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
