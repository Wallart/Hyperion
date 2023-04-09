from time import time
from audio import int16_to_float32
from flask import Flask, Response, request
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import logging
import requests
import numpy as np

app = Flask(__name__)


@app.route('/talk', methods=['POST'])
def handle_audio_stream():
    # wav, _ = librosa.load('/Users/wallart/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav', sr=24000)
    # wav = float32_to_int16(wav)
    # return Response(response=wav.tobytes(), status=200, mimetype='application/octet-stream')
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

    try:
        t0 = time()
        logging.info('Processing request...')
        payload = [
            ('speaker', ('speaker', recognized_speaker, 'text/plain')),
            ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
        ]
        res = requests.post(url=f'{target_url}/audio', files=payload, stream=True)
        if res.status_code != 200:
            return Response(response=f'Something went wrong. HTTP {res.status_code}', status=500, mimetype='text/plain')
        else:
            req = None
            for chunk in res.iter_lines(delimiter=b'----CHUNK-END----\n'):
                sub_chunks = chunk.split(b'----TEXT-END----\n')
                if len(sub_chunks) == 3:
                    req = sub_chunks[0].decode('utf-8')
                    answer = sub_chunks[1].decode('utf-8')
                    audio = sub_chunks[2]
                    print(f'{recognized_speaker} : {req}')
                elif len(sub_chunks) == 2:
                    answer = sub_chunks[0].decode('utf-8')
                    audio = sub_chunks[1]
                else:
                    continue

                print(f'ChatGPT : {answer}')
                spoken_chunk = np.frombuffer(audio, dtype=np.int16)[1000:]  # remove popping sound
                return Response(response=spoken_chunk.tobytes(), status=200, mimetype='application/octet-stream')

        logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
    except Exception as e:
        logging.warning(f'Request canceled : {e}')
        return Response(response=str(e), status=500, mimetype='text/plain')
    return Response(response='OK', status=200, mimetype='text/plain')


SAMPLE_RATE = 16000

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    target_url = 'http://deepbox:9999'

    recognizer = VoiceRecognizer()
    detector = VoiceDetector(SAMPLE_RATE, activation_threshold=.9)

    intake = detector.create_intake()
    sink = detector.pipe(recognizer).create_sink()

    detector.start()
    recognizer.start()

    app.run(host='0.0.0.0', debug=False, threaded=True, port=9999)
