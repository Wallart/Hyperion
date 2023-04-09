#!/usr/bin/env python

from time import time
from utils.logger import ProjectLogger
from utils.request import RequestObject
from utils.utils import get_ctx
from utils.protocol import frame_encode
from analysis.chat_gpt import ChatGPT, CHAT_MODELS
from utils.execution import startup, handle_errors
from analysis.command_detector import CommandDetector, ACTIONS
from flask_log_request_id import RequestID, current_request_id
from voice_processing.voice_synthesizer import VoiceSynthesizer
from flask_socketio import SocketIO, send, emit
from flask import Flask, Response, request, g, stream_with_context
from voice_processing.voice_transcriber import VoiceTranscriber, TRANSCRIPT_MODELS

import argparse
# import eventlet
import numpy as np


class Brain:
    def __init__(self, ctx, opts, host='0.0.0.0'):
        self.host = host
        self.name = opts.name
        self.port = opts.port
        self.debug = opts.debug
        self.frozen = False

        self.transcriber = VoiceTranscriber(ctx, opts.whisper)
        self.chat = ChatGPT(opts.name, opts.gpt, opts.no_memory, opts.clear, opts.prompt)
        self.synthesizer = VoiceSynthesizer()

        self.transcriber.pipe(self.chat).pipe(self.synthesizer)

        # commands handling block
        self.commands = CommandDetector()
        self.transcriber.pipe(self.commands)
        self.cmd_sink = self.commands.create_sink()

        self.audio_intake = self.transcriber.create_intake()
        self.text_intake = self.chat.get_intake()

        self.threads = [self.transcriber, self.commands, self.chat, self.synthesizer]

    def start(self, socket_io, flask_app):
        try:
            _ = [t.start() for t in self.threads]
            # app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
            socket_io.run(flask_app, host=self.host, debug=self.debug, port=self.port)
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

    def handle_audio(self, request_id, speaker, speech):
        request_obj = RequestObject(request_id, speaker)
        request_obj.set_audio_request(speech)

        sink = self.synthesizer.create_identified_sink(request_id)
        self.audio_intake.put(request_obj)

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


APP_NAME = 'hyperion_brain'
app = Flask(__name__)
RequestID(app)
socketio = SocketIO(app)


@socketio.on('connect')
def connect():
    ProjectLogger().info(f'Client {request.sid} connected')


@socketio.on('disconnect')
def disconnect():
    ProjectLogger().info(f'Client {request.sid} disconnected')


@app.route('/name', methods=['GET'])
def name():
    return brain.name, 200


@app.route('/audio', methods=['POST'])
def http_audio_stream():
    request_id = current_request_id()
    speech = request.files['speech'].read()
    speaker = request.files['speaker'].read().decode('utf-8')

    stream = brain.handle_audio(request_id, speaker, speech)

    if brain.frozen:
        return 'I\'m a teapot', 418

    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@socketio.on('audio')
def sio_audio_stream(data):
    request_id = request.sid
    speaker = data['speaker']
    speech = data['speech']

    stream = brain.handle_audio(request_id, speaker, speech)
    for frame in stream:
        emit('answer', dict(requester=speaker, answer=frame), to=request_id)
        socketio.sleep(0)  # force flush all emit calls. Should we import geventlet ?


@app.route('/chat', methods=['POST'])
def http_chat():
    request_id = current_request_id()
    user = request.form['user']
    message = request.form['message']

    if user is None or message is None:
        return 'Invalid chat request', 500

    if '!FREEZE' in message:
        brain.frozen = True
        brain.chat.frozen = True
        return 'Freezed', 202
    elif '!UNFREEZE' in message:
        brain.frozen = False
        brain.chat.frozen = False
        return 'Unfreezed', 202

    stream = brain.handle_chat(request_id, user, message)
    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@socketio.on('chat')
def sio_chat(data):
    request_id = request.sid
    user = data['user']
    message = data['message']

    if user is None or message is None:
        return

    stream = brain.handle_chat(request_id, user, message)
    # _ = [emit('answer', dict(requester=user, answer=frame), to=request_id) for frame in stream]
    for frame in stream:
        emit('answer', dict(requester=user, answer=frame), to=request_id)
        socketio.sleep(0)  # force flush all emit calls. Should we import geventlet ?


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


@handle_errors
def main(args):
    global brain
    ctx = get_ctx(args)
    brain = Brain(ctx, args)
    brain.start(socketio, app)


if __name__ == '__main__':
    def add_opts(sub_parser):
        sub_parser.add_argument('-p', '--port', type=int, default=9999, help='Listening port.')
        sub_parser.add_argument('--clear', action='store_true', help='Clean persistent memory at startup')
        sub_parser.add_argument('--no-memory', action='store_true', help='Start bot without persistent memory.')
        sub_parser.add_argument('--name', type=str, default='Hypérion', help='Set bot name.')
        sub_parser.add_argument('--gpt', type=str, default=CHAT_MODELS[1], choices=CHAT_MODELS, help='GPT version to use.')
        sub_parser.add_argument('--whisper', type=str, default=TRANSCRIPT_MODELS[3], choices=TRANSCRIPT_MODELS, help='Whisper version to use.')
        sub_parser.add_argument('--prompt', type=str, default='base', help='Prompt file to use.')

    parser = argparse.ArgumentParser(description='Hyperion\'s brain')
    parser.add_argument('--debug', action='store_true', help='Enables debugging.')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
    parser.add_argument('--foreground', dest='daemon', action='store_false', help='Run in foreground.')
    sub_parsers = parser.add_subparsers(dest='action', required=True)

    add_opts(sub_parsers.add_parser('start'))
    add_opts(sub_parsers.add_parser('restart'))
    sub_parsers.add_parser('stop')

    startup(APP_NAME.lower(), parser, main)
