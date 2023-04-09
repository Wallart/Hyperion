#!/usr/bin/env python

from time import time
from daemonocle import Daemon
from utils.logger import ProjectLogger
from utils.request import RequestObject
from utils.utils import get_ctx, frame_encode
from analysis.chat_gpt import ChatGPT, CHAT_MODELS
from flask_log_request_id import RequestID, current_request_id
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber
from flask import Flask, Response, request, g, stream_with_context

import os
import argparse


class Brain:
    def __init__(self, ctx, port, debug, name, model, memory, clear, prompt, host='0.0.0.0'):
        self.host = host
        self.port = port
        self.debug = debug

        self.transcriber = VoiceTranscriber(ctx)
        self.chat = ChatGPT(name, model, memory, clear, prompt)
        self.synthesizer = VoiceSynthesizer()

        self.transcriber.pipe(self.chat).pipe(self.synthesizer)

        self.audio_intake = self.transcriber.create_intake()
        self.text_intake = self.chat.get_intake()

        self.threads = [self.transcriber, self.chat, self.synthesizer]

    def boot(self):
        try:
            _ = [t.start() for t in self.threads]
            app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]

    def sink_streamer(self, sink):
        request_id = current_request_id()
        while True:
            request_obj = sink.drain()

            if request_obj.text_answer is None and request_obj.audio_answer is None and request_obj.termination:
                self.synthesizer.delete_identified_sink(request_id)
                return

            if request_obj.audio_answer is None:
                # if text answer not empty but audio is, means speech synthesis api call failed
                ProjectLogger().warning('Speech synthesis failed.')
                continue

            yield frame_encode(request_obj.num_answer, request_obj.text_request, request_obj.text_answer, request_obj.audio_answer)

    def handle_audio(self):
        request_id = current_request_id()
        speech = request.files['speech'].read()
        speaker = request.files['speaker'].read().decode('utf-8')

        request_obj = RequestObject(request_id, speaker)
        request_obj.set_audio_request(speech)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.audio_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return Response(response=stream_with_context(stream), mimetype='application/octet-stream')

    def handle_chat(self):
        request_id = current_request_id()
        user = request.form['user']
        message = request.form['message']

        if user is None or message is None:
            return 'Invalid chat request', 500

        request_obj = RequestObject(request_id, user)
        request_obj.set_text_request(message)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.text_intake.put(request_obj)

        stream = self.sink_streamer(sink)
        return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


APP_NAME = 'hyperion_brain'
app = Flask(__name__)
RequestID(app)


@app.route('/audio', methods=['POST'])
def audio_stream():
    return brain.handle_audio()


@app.route('/chat', methods=['POST'])
def chat():
    return brain.handle_chat()


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
