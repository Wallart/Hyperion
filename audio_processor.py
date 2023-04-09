from time import time
from functools import partial
from audio.audio_file import AudioFile
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from audio.speakers_stream import SpeakersStream
from voice_processing.voice_detector import VoiceDetector
from requests_toolbelt import MultipartDecoder

import requests
import logging
import numpy as np
import sounddevice as sd

from voice_processing.voice_recognizer import VoiceRecognizer

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    OUT_SAMPLE_RATE = 24000
    target_url = 'http://deepbox:9999'
    # audio_clazz = partial(AudioFile, './resources/speakers_samples/stef/stef.wav')
    audio_clazz = partial(Microphone)

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
                res = requests.post(url=f'{target_url}/audio', files=payload)
                if res.status_code != 200:
                    logging.warning(f'Something went wrong. HTTP {res.status_code}')
                else:
                    raw_request, raw_response, raw_audio = MultipartDecoder.from_response(res).parts

                    written_request = raw_request.text
                    written_response = raw_response.text
                    spoken_response = np.frombuffer(raw_audio.content, dtype=np.float32)
                    # sd.stop()
                    # sd.play(spoken_response, blocking=False, samplerate=OUT_SAMPLE_RATE)
                    intake.put(spoken_response)
                    logging.info(written_request)
                    logging.info(f'ChatGPT : {written_response}')

                logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
            except Exception as e:
                logging.warning(f'Request canceled : {e}')
