#!/usr/bin/python

from time import time, sleep
from utils.execution import startup, handle_errors
from utils.logger import ProjectLogger
from utils.utils import get_ctx
from utils.protocol import frame_encode
from flask import Flask, Response, request, g
from audio import int16_to_float32, float32_to_int16
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import os
import librosa
import argparse
import requests
import numpy as np


class NetworkEar:

    def __init__(self, ctx, opts, host='0.0.0.0', sample_rate=16000):
        self.host = host
        self.debug = opts.debug
        self.port = opts.port
        self.target_url = f'http://{opts.target_url}'
        self.dummy_file = opts.dummy_file if opts.dummy_file is None else os.path.expanduser(opts.dummy_file)
        self._no_recog = opts.no_recog

        detector = VoiceDetector(ctx, sample_rate, activation_threshold=.9)
        self.intake = detector.create_intake()

        if opts.no_recog:
            self.sink = detector.create_sink()
            self.threads = [detector]
        else:
            recognizer = VoiceRecognizer(ctx)
            self.sink = detector.pipe(recognizer).create_sink()
            self.threads = [recognizer, detector]

    def boot(self):
        try:
            _ = [t.start() for t in self.threads]
            app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        _ = [t.join() for t in self.threads]

    def dummy_response(self):
        def generator(n):
            for _ in range(n):
                wav, _ = librosa.load(self.dummy_file, sr=24000)
                wav = float32_to_int16(wav)
                yield frame_encode(0, 'Mon chien', 'est le plus beau', wav)
                sleep(1)

        return Response(response=generator(1), mimetype='application/octet-stream')

    def chat_gpt_response(self):
        buffer = bytearray()
        for bytes_chunk in request.stream:
            buffer.extend(bytes_chunk)

        buffer = np.frombuffer(buffer, dtype=np.int16)
        buffer = int16_to_float32(buffer)

        self.intake.put(buffer)
        self.intake.put(None)  # end of speech

        data = self.sink.drain()
        audio_chunk, recognized_speaker = data, 'Unknown'
        if not self._no_recog:
            audio_chunk, recognized_speaker = data
        audio_chunk = audio_chunk.numpy()  # TODO Convert to int16
        # if recognized_speaker == 'Unknown':
        #     return Response(response='Unknown speaker', status=204, mimetype='text/plain')

        t0 = time()
        ProjectLogger().info('Processing audio...')
        payload = [
            ('speaker', ('speaker', recognized_speaker, 'text/plain')),
            ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
        ]
        try:
            res = requests.post(url=f'{self.target_url}/audio', files=payload, stream=True)
            ProjectLogger().info(f'Audio processed in {time() - t0:.3f} sec(s).')
            if res.status_code != 200:
                return Response(response=f'Something went wrong. HTTP {res.status_code}', status=500, mimetype='text/plain')
            return Response(response=res, mimetype='application/octet-stream')
        except Exception as e:
            ProjectLogger().warning(f'Request canceled : {e}')
            return Response(response=str(e), status=500, mimetype='text/plain')

    def handle(self):
        if self.dummy_file:
            return self.dummy_response()
        else:
            return self.chat_gpt_response()


APP_NAME = 'hyperion_network_ear'
app = Flask(__name__)


@app.route('/talk', methods=['POST'])
def talk():
    return ear.handle()


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
    global ear
    ctx = get_ctx(args)
    ear = NetworkEar(ctx, args)
    ear.boot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s network ear')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon')
    parser.add_argument('--debug', action='store_true', help='Enables debugging.')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')

    parser.add_argument('-p', '--port', type=int, default=9998, help='Listening port')
    parser.add_argument('-t', '--target-url', type=str, default='localhost:9999', help='Brain target URL')
    parser.add_argument('--dummy-file', type=str, help='Play file instead of Brain\'s responses')
    parser.add_argument('--no-recog', action='store_true', help='Start bot without user recognition.')

    startup(APP_NAME.lower(), parser, main)
