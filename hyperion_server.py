#!/usr/bin/env python
from pathlib import Path
from flask_cors import CORS
from base64 import b64decode
from time import time, sleep
from Crypto.Signature import pss
from Crypto.PublicKey import RSA
from hyperion.utils import get_ctx
from Crypto.Hash import SHA256, SHA1
from Crypto.Cipher import PKCS1_OAEP
from flask_socketio import SocketIO, emit
from hyperion.analysis import CHAT_MODELS
from werkzeug.utils import secure_filename
from hyperion.pipelines.brain import Brain
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager
from hyperion.utils.memory_utils import MANAGER_TOKEN
from hyperion.utils.identity_store import IdentityStore
from hyperion.analysis.prompt_manager import PromptManager
from hyperion.utils.execution import startup, handle_errors
from flask_log_request_id import RequestID, current_request_id
from flask import Flask, Response, request, g, stream_with_context
from hyperion.voice_processing.voice_synthesizer import VALID_ENGINES
from hyperion.voice_processing.voice_transcriber import TRANSCRIPT_MODELS
from hyperion import HYPERION_VERSION, THEIA_MIN_VERSION, HYPERION_RAW_SECRET

import os
import io
import json
import argparse


APP_NAME = os.path.basename(__file__).split('.')[0]
app = Flask(__name__)
CORS(app)
RequestID(app)
sio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')
# Load private key
private_key = RSA.import_key(open(ProjectPaths().resources_dir / 'secret' / 'private_key.pem').read())
cipher_rsa = PKCS1_OAEP.new(private_key, hashAlgo=SHA256)


def get_headers_params():
    request_sid = request.headers['SID']
    preprompt = request.headers['preprompt'] if 'preprompt' in request.headers else None
    llm = request.headers['model'] if 'model' in request.headers else None
    speech_engine = request.headers['speech_engine'] if 'speech_engine' in request.headers else None
    voice = request.headers['voice'] if 'voice' in request.headers else None
    silent = json.loads(request.headers['silent'].lower()) if 'silent' in request.headers else False
    indexes = request.headers['indexes'].split(',') if 'indexes' in request.headers else []
    return request_sid, preprompt, llm, speech_engine, voice, silent, indexes


@sio.on('connect')
def connect():
    ProjectLogger().info(f'Client {request.sid} connected')
    IdentityStore()[request.sid] = None


@sio.on('disconnect')
def disconnect():
    ProjectLogger().info(f'Client {request.sid} disconnected')
    del IdentityStore()[request.sid]


@sio.on('identify')
def on_identify(identity):
    IdentityStore()[request.sid] = identity


@app.route('/version', methods=['GET'])
def version():
    return HYPERION_VERSION, 200


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
    return brain.voice_synthesizer.get_preferred_engines(), 200


@app.route('/tts-preferred-engines', methods=['PUT'])
def set_preferred_engines():
    res = brain.voice_synthesizer.set_preferred_engines(request.json)
    if not res:
        return 'Invalid ordering', 400
    return 'TTS engines order changed', 200


@app.route('/voices', methods=['GET'])
def get_voices():
    engine = request.args.get('engine')
    res = brain.voice_synthesizer.get_engine_valid_voices(engine)
    if not res:
        return [], 200
    return res, 200


@app.route('/voice', methods=['GET'])
def get_voice():
    engine = request.args.get('engine')
    res = brain.voice_synthesizer.get_engine_default_voice(engine)
    if not res:
        return '', 200
    return res, 200


@app.route('/voice', methods=['PUT'])
def set_voice():
    voice = request.form['voice']
    engine = request.form['engine']
    res = brain.voice_synthesizer.set_engine_default_voice(engine, voice)
    if not res:
        return 'Invalid engine and/or voice', 400
    return f'{voice} set for engine {engine}', 200


@app.route('/models', methods=['GET'])
def list_models():
    return list(CHAT_MODELS.keys()), 200


@app.route('/model', methods=['GET'])
def get_model():
    return brain.chat_gpt.get_model(), 200


@app.route('/model', methods=['PUT'])
def set_model():
    model = request.form['model']
    if not brain.chat_gpt.set_model(model):
        return f'{model} prompt not found', 404

    return 'Default model changed', 200


  #################
 # PROMPT routes #
#################
@app.route('/prompts', methods=['GET'])
def list_prompts():
    return PromptManager.list_prompts(), 200


@app.route('/prompts', methods=['POST'])
def upload_prompts():
    if len(request.files) == 0:
        return 'No file(s) found.', 400

    save_count = brain.chat_gpt.prompt_manager.save_prompts(request.files.to_dict())
    return f'{save_count} prompt(s) saved', 200


@app.route('/prompt', methods=['GET'])
def get_prompt():
    return brain.chat_gpt.prompt_manager.get_prompt(), 200


@app.route('/prompt', methods=['PUT'])
def set_prompt():
    prompt = request.form['prompt']
    if not brain.chat_gpt.prompt_manager.set_prompt(prompt):
        return f'{prompt} prompt not found', 404

    return 'Default prompt changed', 200


@app.route('/prompt/<string:prompt_name>', methods=['GET'])
def read_prompt(prompt_name):
    try:
        prompt_content = PromptManager.read_prompt(prompt_name)
        return ''.join(prompt_content), 200
    except Exception:
        return f'Unable to read prompt {prompt_name}', 500


@app.route('/prompt/<string:prompt_name>', methods=['DELETE'])
def delete_prompt(prompt_name):
    deleted = brain.chat_gpt.prompt_manager.delete_prompt(prompt_name)
    if deleted:
        return f'{prompt_name} has been deleted', 200
    return f'{prompt_name} not found', 400


  ################
 # OTHER ROUTES #
################
@app.route('/speech', methods=['POST'])
def http_speech_stream():
    request_id = current_request_id()
    request_sid, preprompt, llm, speech_engine, voice, silent, indexes = get_headers_params()

    speech = request.files['speech'].read()
    speaker = request.files['speaker'].read().decode('utf-8')

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm, speech_engine, voice, silent, indexes)

    if brain.user_commands.frozen:
        return 'I\'m a teapot', 418

    return Response(response=stream_with_context(stream), mimetype='application/octet-stream')


@app.route('/audio', methods=['POST'])
def http_audio_stream():
    request_id = current_request_id()
    request_sid, preprompt, llm, speech_engine, voice, silent, indexes = get_headers_params()

    audio = request.files['audio'].read() if 'audio' in request.files else request.data

    speaker, speech = brain.handle_audio(audio)
    if speaker is None and speech is None:
        return 'No speech detected', 204

    stream = brain.handle_speech(request_id, request_sid, speaker, speech, preprompt, llm, speech_engine, voice, silent, indexes)

    if brain.user_commands.frozen:
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
    request_sid, preprompt, llm, speech_engine, voice, silent, indexes = get_headers_params()

    user = request.form['user']
    message = request.form['message']

    if user is None or message is None:
        return 'Invalid chat request', 500

    # TODO Legacy to be removed
    if '!FREEZE' in message:
        brain.user_commands.frozen = True
        return 'Freezed', 202
    elif '!UNFREEZE' in message:
        brain.user_commands.frozen = False
        return 'Unfreezed', 202

    stream = brain.handle_chat(request_id, request_sid, user, message, preprompt, llm, speech_engine, voice, silent, indexes)
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
    frame = request.files['frame'].read()
    caption = brain.handle_frame(frame)
    return caption, 200


  ##################
 # INDEXES routes #
##################
@app.route('/indexes', methods=['GET'])
def list_indexes():
    response = memoryManager.list_indexes()
    return response._getvalue(), 200


@app.route('/index/state', methods=['GET'])
def memory_state():
    status = 'offline'
    try:
        status = memoryManager.get_status()._getvalue()
    except Exception:
        pass
    return status, 200


@app.route('/index/<string:index>', methods=['POST'])
def create_index(index):
    _ = memoryManager.create_empty_index(index)
    return f'Index {index} created', 200


@app.route('/index/<string:index>', methods=['DELETE'])
def delete_index(index):
    _ = memoryManager.delete_index(index)
    return f'Index {index} deleted', 200


@app.route('/index/<string:index>/documents', methods=['GET'])
def list_documents(index):
    response = memoryManager.list_documents(index)
    return response._getvalue(), 200


@app.route('/index/<string:index>/documents/<string:doc_id>', methods=['DELETE'])
def delete_from_index(index, doc_id):
    _ = memoryManager.delete_from_index(index, doc_id)
    return f'{doc_id} deleted from index {index}', 200


@app.route('/index/<string:index>/query', methods=['GET'])
def query_index(index):
    query_value = request.args.get('value', None)
    if query_value is None:
        return 'Missing query param', 400

    response = memoryManager.query_index(index, query_value)
    return str(response._getvalue()), 200


@app.route('/index/<string:index>/upload', methods=['POST'])
def upload_file_to_index(index):
    if len(request.files) == 0:
        return 'No file(s) found.', 400

    upload_dir = Path('/') / 'tmp' / 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    for fileindex, uploaded_file in request.files.items():
        try:
            filename = secure_filename(uploaded_file.filename)
            filepath = upload_dir / filename
            uploaded_file.save(filepath)

            memoryManager.insert_into_index(index, filename, str(filepath))
        except Exception as e:
            return f'File upload failed. {str(e)}', 500

    return 'File(s) indexed.', 200


@app.route('/upload-to-context', methods=['POST'])
def upload_file_to_context():
    if len(request.files) == 0:
        return 'No file(s) found.', 400

    request_sid, preprompt, _, _, _, _, _ = get_headers_params()

    files_added = 0
    for k, v in request.files.items():
        if v.mimetype != 'application/pdf':
            ProjectLogger().error(f'{v.mimetype} not supported.')
            continue

        brain.handle_document(io.BytesIO(v.read()), preprompt)
        files_added += 1

    if files_added > 0:
        return 'File(s) added to context.', 200
    else:
        return 'No file added. Invalid format', 500


def secret_check():
    try:
        encrypted_secret = request.headers.get('secret')
        decrypted_secret = cipher_rsa.decrypt(b64decode(encrypted_secret)).decode('utf-8')
        if decrypted_secret == HYPERION_RAW_SECRET:
            return True
    except Exception as e:
        pass
    return False


@app.before_request
def before_request():
    g.start = time()

    # Don't block pre-flight requests.
    if request.method != 'OPTIONS':
        client_version = request.headers.get('version')
        if client_version is None:
            return 'Client update required', 426  # HTTP : Upgrade required

        min_version = THEIA_MIN_VERSION.split('.')
        client_version = client_version.split('.')
        valid = [int(acc_vers) <= int(client_vers) for acc_vers, client_vers in zip(min_version, client_version)]
        if not all(valid):
            return 'Client update required', 426  # HTTP : Upgrade required

        if not secret_check():
            ProjectLogger().warning(f'Invalid secret for request {request.method} {request.base_url}')
            return 'Invalid secret', 401  # HTTP : Unauthorized


@app.after_request
def after_request(response):
    diff = time() - g.start
    ProjectLogger().info(f'Request execution time {diff:.3f} sec(s)')
    return response


@handle_errors
def main(args):
    global brain
    global memoryManager

    memoryManager = BaseManager(('', 5602), bytes(MANAGER_TOKEN, encoding='utf8'))
    memoryManager.register('get_status')
    memoryManager.register('list_indexes')
    memoryManager.register('create_empty_index')
    memoryManager.register('query_index')
    memoryManager.register('delete_index')
    memoryManager.register('insert_into_index')
    memoryManager.register('delete_from_index')
    memoryManager.register('list_documents')
    while True:
        try:
            memoryManager.connect()
            break
        except Exception:
            timeout = 10
            ProjectLogger().info(f'Memory server not yet started. Waiting {timeout} sec(s)...')
            sleep(timeout)

    ctx = get_ctx(args)
    brain = Brain(ctx, args)
    brain.start(sio, app)


if __name__ == '__main__':
    def add_opts(sub_parser):
        sub_parser.add_argument('-p', '--port', type=int, default=9999, help='Listening port.')
        sub_parser.add_argument('--llama-host', type=str, default='localhost', help='Llama server host')
        sub_parser.add_argument('--llama-port', type=int, default=8080, help='Llama server port')
        sub_parser.add_argument('--clear', action='store_true', help='Clean persistent memory at startup')
        sub_parser.add_argument('--no-memory', action='store_true', help='Start bot without persistent memory.')
        sub_parser.add_argument('--name', type=str, default='Hypérion', help='Set bot name.')
        sub_parser.add_argument('--gpt', type=str, default=list(CHAT_MODELS.keys())[0], choices=CHAT_MODELS.keys(), help='GPT version to use.')
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
