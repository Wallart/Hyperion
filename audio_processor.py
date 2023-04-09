from time import time
from functools import partial
from audio.audio_file import AudioFile
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from audio.speakers_stream import SpeakersStream
from voice_processing.voice_detector import VoiceDetector

import requests
import logging
import numpy as np
import sounddevice as sd

from voice_processing.voice_recognizer import VoiceRecognizer

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    OUT_SAMPLE_RATE = 24000
    target_url = 'http://deepbox:9999'
    # audio_clazz = partial(AudioFile, '~/datasets/test.wav')
    audio_clazz = partial(Microphone, device_idx=0)

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
            audio_chunk, recognized = sink.get()
            audio_chunk = audio_chunk.numpy()
            if not recognized:
                continue

            try:
                t0 = time()
                logging.info('Processing request...')
                res = requests.post(url=f'{target_url}/audio', data=audio_chunk.tobytes(), headers={'Content-Type': 'application/octet-stream'})
                if res.status_code != 200:
                    logging.warning(f'Something went wrong. HTTP {res.status_code}')
                else:
                    spoken_response = np.frombuffer(res.content, dtype=np.float32)
                    # sd.stop()
                    # sd.play(spoken_response, blocking=False, samplerate=OUT_SAMPLE_RATE)
                    intake.put(spoken_response)
                logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
            except Exception as e:
                logging.warning(f'Request canceled : {e}')
