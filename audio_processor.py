from time import time
from functools import partial
from audio.audio_file import AudioFile
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from audio.speakers_stream import SpeakersStream
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import requests
import logging
import numpy as np
import sounddevice as sd


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    OUT_SAMPLE_RATE = 24000
    target_url = 'http://deepbox:9999'
    audio_clazz = partial(Microphone)
    # audio_clazz = partial(AudioFile, './resources/speakers_samples/stef/stef.wav')

    with audio_clazz(duration_ms=512) as source:
        stream = AudioStream(source)
        detector = VoiceDetector(source.sample_rate(), activation_threshold=.9)
        recognizer = VoiceRecognizer()
        speakers = SpeakersStream(OUT_SAMPLE_RATE, source.channels())

        sink = stream.pipe(detector).pipe(recognizer).create_sink()
        intake = speakers.create_intake()

        stream.start()
        detector.start()
        recognizer.start()
        speakers.start()

        while True:
            audio_chunk, recognized_speaker = sink.get()
            audio_chunk = audio_chunk.numpy()
            if recognized_speaker == 'Unknown':
                continue

            try:
                t0 = time()
                logging.info('Processing request...')
                payload = [
                    ('speaker', ('speaker', recognized_speaker, 'text/plain')),
                    ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
                ]
                res = requests.post(url=f'{target_url}/audio', files=payload, stream=True)
                if res.status_code != 200:
                    logging.warning(f'Something went wrong. HTTP {res.status_code}')
                else:
                    request = None
                    for chunk in res.iter_lines(delimiter=b'----CHUNK-END----\n'):
                        sub_chunks = chunk.split(b'----TEXT-END----\n')
                        if len(sub_chunks) == 3:
                            request = sub_chunks[0].decode('utf-8')
                            answer = sub_chunks[1].decode('utf-8')
                            audio = sub_chunks[2]
                            print(f'{recognized_speaker} : {request}')
                        elif len(sub_chunks) == 2:
                            answer = sub_chunks[0].decode('utf-8')
                            audio = sub_chunks[1]
                        else:
                            continue

                        print(f'ChatGPT : {answer}')
                        spoken_chunk = np.frombuffer(audio, dtype=np.float32)
                        intake.put(spoken_chunk)

                logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
            except Exception as e:
                logging.warning(f'Request canceled : {e}')
