from time import time
from flask import Flask, Response, request
from audio import int16_to_float32, float32_to_int16
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import logging
import librosa
import requests
import numpy as np

app = Flask(__name__)

TEXT_SEPARATOR = b'----TEXT-END----\n'
CHUNK_SEPARATOR = b'----CHUNK-END----\n'

SAMPLE_RATE = 16000
TARGET_URL = 'http://deepbox:9999'


def dummy_response():
    wav, _ = librosa.load('/Users/wallart/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav', sr=24000)
    wav = float32_to_int16(wav)
    part1 = bytes('TOTO', 'utf-8') + TEXT_SEPARATOR + bytes('TATA', 'utf-8') + TEXT_SEPARATOR + wav.tobytes() + CHUNK_SEPARATOR
    part2 = bytes('TUTU', 'utf-8') + TEXT_SEPARATOR + bytes('TITI', 'utf-8') + TEXT_SEPARATOR + wav[:20000].tobytes() + CHUNK_SEPARATOR
    return Response(response=part1 + part2, status=200, mimetype='application/octet-stream')


def chat_gpt_response():
    buffer = bytearray()
    for bytes_chunk in request.stream:
        buffer.extend(bytes_chunk)

    buffer = np.frombuffer(buffer, dtype=np.int16)
    buffer = int16_to_float32(buffer)

    intake.put(buffer)
    intake.put(None)  # end of speech
    audio_chunk, recognized_speaker = sink.get()
    audio_chunk = audio_chunk.numpy()
    # if recognized_speaker == 'Unknown':
    #     return Response(response='Unknown speaker', status=204, mimetype='text/plain')

    t0 = time()
    logging.info('Processing request...')
    payload = [
        ('speaker', ('speaker', recognized_speaker, 'text/plain')),
        ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
    ]
    try:
        res = requests.post(url=f'{TARGET_URL}/audio', files=payload, stream=True)
        logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
        if res.status_code != 200:
            return Response(response=f'Something went wrong. HTTP {res.status_code}', status=500, mimetype='text/plain')
        return Response(response=res, status=200, mimetype='application/octet-stream')
    except Exception as e:
        logging.warning(f'Request canceled : {e}')
        return Response(response=str(e), status=500, mimetype='text/plain')


@app.route('/talk', methods=['POST'])
def handle_audio_stream():
    # return dummy_response()
    return chat_gpt_response()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    recognizer = VoiceRecognizer()
    detector = VoiceDetector(SAMPLE_RATE, activation_threshold=.9)

    intake = detector.create_intake()
    sink = detector.pipe(recognizer).create_sink()

    detector.start()
    recognizer.start()

    app.run(host='0.0.0.0', debug=False, threaded=True, port=9999)
