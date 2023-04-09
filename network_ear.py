#!/usr/bin/python

from time import time
from daemonocle import Daemon
from flask import Flask, Response, request
from audio import int16_to_float32, float32_to_int16
from utils import TEXT_SEPARATOR, CHUNK_SEPARATOR
from utils.logger import ProjectLogger
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import os
import librosa
import argparse
import requests
import numpy as np


class NetworkEar:

    def __init__(self, port, debug, target_url, host='0.0.0.0', dummy_file=None, sample_rate=16000):
        self.host = host
        self.debug = debug
        self.port = port
        self.target_url = f'http://{target_url}'
        self.dummy_file = dummy_file if dummy_file is None else os.path.expanduser(dummy_file)

        self.recognizer = VoiceRecognizer()
        self.detector = VoiceDetector(sample_rate, activation_threshold=.9)

        self.intake = self.detector.create_intake()
        self.sink = self.detector.pipe(self.recognizer).create_sink()

    def boot(self):
        self.detector.start()
        self.recognizer.start()

        app.run(host=self.host, debug=self.debug, threaded=True, port=self.port)

    def dummy_response(self):
        wav, _ = librosa.load(self.dummy_file, sr=24000)
        wav = float32_to_int16(wav)
        part1 = bytes('TOTO', 'utf-8') + TEXT_SEPARATOR + bytes('TATA', 'utf-8') + TEXT_SEPARATOR + wav.tobytes() + CHUNK_SEPARATOR
        part2 = bytes('TUTU', 'utf-8') + TEXT_SEPARATOR + bytes('TITI', 'utf-8') + TEXT_SEPARATOR + wav[:20000].tobytes() + CHUNK_SEPARATOR
        return Response(response=part1 + part2, status=200, mimetype='application/octet-stream')

    def chat_gpt_response(self):
        buffer = bytearray()
        for bytes_chunk in request.stream:
            buffer.extend(bytes_chunk)

        buffer = np.frombuffer(buffer, dtype=np.int16)
        buffer = int16_to_float32(buffer)

        self.intake.put(buffer)
        self.intake.put(None)  # end of speech
        audio_chunk, recognized_speaker = self.sink.get()
        audio_chunk = audio_chunk.numpy()
        # if recognized_speaker == 'Unknown':
        #     return Response(response='Unknown speaker', status=204, mimetype='text/plain')

        t0 = time()
        ProjectLogger().info('Processing request...')
        payload = [
            ('speaker', ('speaker', recognized_speaker, 'text/plain')),
            ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
        ]
        try:
            res = requests.post(url=f'{self.target_url}/audio', files=payload, stream=True)
            ProjectLogger().info(f'Request processed in {time() - t0:.3f} sec(s).')
            if res.status_code != 200:
                return Response(response=f'Something went wrong. HTTP {res.status_code}', status=500, mimetype='text/plain')
            return Response(response=res, status=200, mimetype='application/octet-stream')
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


def main():
    try:
        global ear
        ear = NetworkEar(args.port, args.debug, args.target_url, dummy_file=args.dummy_file)
        ear.boot()
    except Exception as e:
        ProjectLogger().error(f'Fatal error occurred : {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s brain')
    parser.add_argument('-p', '--port', type=int, default=9998, help='Listening port')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--target-url', type=str, default='localhost:9999', help='Brain target URL')
    parser.add_argument('--dummy-file', type=str, help='Play file instead of Brain\'s responses')
    parser.add_argument('--debug', action='store_true', help='Enables flask debugging')
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
