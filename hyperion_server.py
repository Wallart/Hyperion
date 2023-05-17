#!/usr/bin/env python
from time import time
from uuid import uuid4
from pathlib import Path
from flask_cors import CORS
from werkzeug.utils import secure_filename
from hyperion.utils.utils import get_ctx
from hyperion.pipelines.brain import Brain
from flask_socketio import SocketIO, emit
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager
from hyperion.analysis.chat_gpt import CHAT_MODELS
from hyperion.analysis.prompt_manager import PromptManager
from hyperion.utils.execution import startup, handle_errors
from flask_log_request_id import RequestID, current_request_id

from hyperion.voice_processing.voice_synthesizer import VALID_ENGINES
from hyperion.voice_processing.voice_transcriber import TRANSCRIPT_MODELS
from flask import Flask, Response, request, g, stream_with_context

import os
import argparse


APP_NAME = os.path.basename(__file__).split('.')[0]
app = Flask(__name__)
CORS(app)
RequestID(app)
sio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')


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


@app.route('/tts-engines', methods=['GET'])
def get_tts_engines():
    return VALID_ENGINES, 200


@app.route('/tts-preferred-engines', methods=['GET'])
def get_preferred_engines():
    return brain.synthesizer.get_preferred_engines(), 200


@app.route('/tts-preferred-engines', methods=['POST'])
def set_preferred_engines():
    res = brain.synthesizer.set_preferred_engines(request.json)
    if not res:
        return 'Invalid ordering', 400
    return 'TTS engines order changed', 200


@app.route('/voices', methods=['GET'])
def get_voices():
    engine = request.form['engine']
    res = brain.synthesizer.get_engine_valid_voices(engine)
    if not res:
        return 'Invalid engine', 400
    return res, 200


@app.route('/voice', methods=['GET'])
def get_voice():
    engine = request.form['engine']
    res = brain.synthesizer.get_engine_default_voice(engine)
    if not res:
        return 'Invalid engine', 400
    return res, 200


@app.route('/voice', methods=['POST'])
def set_voice():
    voice = request.form['voice']
    engine = request.form['engine']
    res = brain.synthesizer.set_engine_default_voice(engine, voice)
    if not res:
        return 'Invalid engine and/or voice', 400
    return f'{voice} set for engine {engine}', 200


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
    speech_engine = request.headers['speech_engine'] if 'speech_engine' in request.headers else None
    voice = request.headers['voice'] if 'voice' in request.headers else None

    speech = request.files['speech'].read()
    speaker = request.files['speaker'].read().decode('utf-8')

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm, speech_engine, voice)

    if brain.frozen:
        return 'I\'m a teapot', 418

    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@app.route('/audio', methods=['POST'])
def http_audio_stream():
    request_id = current_request_id()
    request_sid = request.headers['SID']
    preprompt = request.headers['preprompt'] if 'preprompt' in request.headers else None
    llm = request.headers['model'] if 'model' in request.headers else None
    speech_engine = request.headers['speech_engine'] if 'speech_engine' in request.headers else None
    voice = request.headers['voice'] if 'voice' in request.headers else None

    audio = request.files['audio'].read() if 'audio' in request.files else request.data

    speaker, speech = brain.handle_audio(audio)
    if speaker is None and speech is None:
        return 'No speech detected', 204

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm, speech_engine, voice)

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
    speech_engine = request.headers['speech_engine'] if 'speech_engine' in request.headers else None
    voice = request.headers['voice'] if 'voice' in request.headers else None

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

    stream = brain.handle_chat(request_id, request_sid, user, message, preprompt, llm, speech_engine, voice)
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


@app.route('/query', methods=['GET'])
def query_knowledge_base():
    index_name = request.args.get('index', None)
    query_value = request.args.get('value', None)
    if query_value is None or index_name is None:
        return 'Missing index or query param', 400

    response = memoryManager.query_index(index_name, query_value)
    return str(response._getvalue()), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    index_name = request.form['index_name']
    if len(request.files) == 0:
        return 'No file(s) found.', 400

    upload_dir = Path('/') / 'tmp' / 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    for fileindex, uploaded_file in request.files.items():
        filepath = None
        try:
            # filename = secure_filename(uploaded_file.filename)
            filepath = upload_dir / str(uuid4())
            uploaded_file.save(filepath)

            # if request.form.get('filename_as_doc_id', None) is not None:
            #     manager.insert_into_index(filepath, doc_id=filename)
            # else:
            memoryManager.insert_into_index(index_name, str(filepath))
        except Exception as e:
            return f'File upload failed. {str(e)}', 500
        finally:
            if filepath is not None and filepath.exists():
                os.remove(filepath)

    return 'File(s) indexed.', 200


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
    global memoryManager

    # memoryManager = BaseManager(('', 5602), b'password')
    # memoryManager.register('query_index')
    # memoryManager.register('insert_into_index')
    # memoryManager.connect()

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
