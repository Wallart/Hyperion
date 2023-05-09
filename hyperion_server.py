#!/usr/bin/env python

from time import time
from hyperion.utils.utils import get_ctx
from hyperion.pipelines.brain import Brain
from hyperion.utils.logger import ProjectLogger
from flask_socketio import SocketIO, emit
from hyperion.analysis.chat_gpt import CHAT_MODELS
from hyperion.analysis.prompt_manager import PromptManager
from hyperion.utils.execution import startup, handle_errors
from flask_log_request_id import RequestID, current_request_id
from hyperion.voice_processing.voice_transcriber import TRANSCRIPT_MODELS
from flask import Flask, Response, request, g, stream_with_context

import os
import argparse


APP_NAME = os.path.basename(__file__).split('.')[0]
app = Flask(__name__)
RequestID(app)
sio = SocketIO(app, async_mode='threading')


@sio.on('connect')
def connect():
    ProjectLogger().info(f'Client {request.sid} connected')


@sio.on('disconnect')
def disconnect():
    ProjectLogger().info(f'Client {request.sid} disconnected')


@app.route('/state', methods=['GET'])
def state():
    return 'Up and running', 200


@app.route('/name', methods=['GET'])
def name():
    return brain.name, 200


@app.route('/models', methods=['GET'])
def list_models():
    return CHAT_MODELS, 200


@app.route('/model', methods=['GET'])
def get_model():
    return brain.chat.get_model(), 200


@app.route('/model', methods=['POST'])
def set_model():
    model = request.form['model']
    if not brain.chat.set_model(model):
        return f'{model} prompt not found', 404

    return 'Default model changed', 200


@app.route('/prompts', methods=['GET'])
def list_prompts():
    return PromptManager.list_prompts(), 200


@app.route('/prompt', methods=['GET'])
def get_prompt():
    return brain.chat.prompt_manager.get_prompt(), 200


@app.route('/prompt', methods=['POST'])
def set_prompt():
    prompt = request.form['prompt']
    if not brain.chat.prompt_manager.set_prompt(prompt):
        return f'{prompt} prompt not found', 404

    return 'Default prompt changed', 200


@app.route('/speech', methods=['POST'])
def http_speech_stream():
    request_id = current_request_id()
    request_sid = request.headers['SID']
    preprompt = request.headers['preprompt'] if 'preprompt' in request.headers else None
    llm = request.headers['model'] if 'model' in request.headers else None

    speech = request.files['speech'].read()
    speaker = request.files['speaker'].read().decode('utf-8')

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm)

    if brain.frozen:
        return 'I\'m a teapot', 418

    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@app.route('/audio', methods=['POST'])
def http_audio_stream():
    request_id = current_request_id()
    request_sid = request.headers['SID']
    preprompt = request.headers['preprompt'] if 'preprompt' in request.headers else None
    llm = request.headers['model'] if 'model' in request.headers else None

    audio = request.files['audio'].read() if 'audio' in request.files else request.data

    speaker, speech = brain.handle_audio(audio)
    if speaker is None and speech is None:
        return 'No speech detected', 204

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm)

    if brain.frozen:
        return 'I\'m a teapot', 418

    res = Response(response=stream_with_context(stream), mimetype='application/octet-stream')
    res.headers.add('Speaker', speaker)  # TODO Ugly should be added in communication protocol

    return res


@sio.on('speech')
def sio_speech_stream(data):
    request_id = request.sid
    speaker = data['speaker']
    speech = data['speech']

    stream = brain.handle_speech(request_id, request_id, speaker, speech)
    for frame in stream:
        emit('answer', dict(requester=speaker, answer=frame), to=request_id)
        sio.sleep(0)  # force flush all emit calls. Should we import geventlet ?


@sio.on('audio')
def sio_audio_stream(audio):
    request_id = request.sid

    speaker, speech = brain.handle_audio(audio)
    if speaker is None and speech is None:
        return

    stream = brain.handle_speech(request_id, request_id, speaker, speech)
    for frame in stream:
        emit('answer', dict(requester=speaker, answer=frame), to=request_id)
        sio.sleep(0)  # force flush all emit calls. Should we import geventlet ?


@app.route('/chat', methods=['POST'])
def http_chat():
    request_id = current_request_id()
    request_sid = request.headers['SID']
    preprompt = request.headers['preprompt'] if 'preprompt' in request.headers else None
    llm = request.headers['model'] if 'model' in request.headers else None

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

    stream = brain.handle_chat(request_id, request_sid, user, message, preprompt, llm)
    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@sio.on('chat')
def sio_chat(data):
    request_id = request.sid
    user = data['user']
    message = data['message']

    if user is None or message is None:
        return

    stream = brain.handle_chat(request_id, request_id, user, message)
    # _ = [emit('answer', dict(requester=user, answer=frame), to=request_id) for frame in stream]
    for frame in stream:
        emit('answer', dict(requester=user, answer=frame), to=request_id)
        sio.sleep(0)  # force flush all emit calls. Should we import geventlet ?


@app.route('/video', methods=['POST'])
def video_stream():
    width = int(request.headers['framewidth'])
    height = int(request.headers['frameheight'])
    channels = int(request.headers['framechannels'])
    frame = request.files['frame'].read()

    brain.handle_frame(frame, width, height, channels)
    return 'Frame processed', 200


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
    brain.start(sio, app)


if __name__ == '__main__':
    def add_opts(sub_parser):
        sub_parser.add_argument('-p', '--port', type=int, default=9999, help='Listening port.')
        sub_parser.add_argument('--clear', action='store_true', help='Clean persistent memory at startup')
        sub_parser.add_argument('--no-memory', action='store_true', help='Start bot without persistent memory.')
        sub_parser.add_argument('--name', type=str, default='Hypérion', help='Set bot name.')
        sub_parser.add_argument('--gpt', type=str, default=CHAT_MODELS[0], choices=CHAT_MODELS, help='GPT version to use.')
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
