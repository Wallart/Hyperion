from time import time
from functools import partial
from daemonocle import Daemon
from utils.utils import get_ctx, frame_decode
from audio.audio_file import AudioFile
from utils.logger import ProjectLogger
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from audio.speakers_stream import SpeakersStream
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import os
import argparse
import requests
import logging
import numpy as np


class LocalEar:

    def __init__(self, ctx, target_url, device_idx, out_sample_rate=24000, dummy_file=None):
        self._ctx = ctx
        self.target_url = f'http://{target_url}'
        self.dummy_file = dummy_file if dummy_file is None else os.path.expanduser(dummy_file)

        audio_clazz = partial(Microphone, device_idx=device_idx)
        if self.dummy_file is not None:
            audio_clazz = partial(AudioFile, self.dummy_file)

        self.source = audio_clazz(duration_ms=512)
        self.stream = AudioStream(self.source)
        self.detector = VoiceDetector(ctx, self.source.sample_rate(), activation_threshold=.9)
        self.recognizer = VoiceRecognizer(ctx)
        self.speakers = SpeakersStream(out_sample_rate, self.source.channels())

        self.sink = self.stream.pipe(self.detector).pipe(self.recognizer).create_sink()
        self.intake = self.speakers.create_intake()

    def boot(self):
        self.source.start()

        self.stream.start()
        self.detector.start()
        self.recognizer.start()
        self.speakers.start()

        self.mainloop()

    def mainloop(self):
        while True:
            audio_chunk, recognized_speaker = self.sink.get()
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
                res = requests.post(url=f'{self.target_url}/audio', files=payload, stream=True)
                if res.status_code != 200:
                    logging.warning(f'Something went wrong. HTTP {res.status_code}')
                else:
                    response = bytearray(res.content)
                    while len(response) > 0:
                        decoded_frame, response = frame_decode(response)

                        answer = decoded_frame['ANS']
                        audio = decoded_frame['PCM']

                        print(f'ChatGPT : {answer}')
                        spoken_chunk = np.frombuffer(audio, dtype=np.int16)[1000:]  # remove popping sound
                        # TODO SPOKEN CHUNK MUST BE SENT TO MICROPHONE FOR SUBSTRACTION
                        # stream._audio_source.set_feedback(spoken_chunk, OUT_SAMPLE_RATE)
                        self.intake.put(spoken_chunk)

                logging.info(f'Request processed in {time() - t0:.3f} sec(s).')
            except Exception as e:
                logging.warning(f'Request canceled : {e}')


APP_NAME = 'hyperion_local_ear'


def main():
    try:
        ctx = get_ctx(args)
        ear = LocalEar(ctx, args.target_url, device_idx=args.device_idx, dummy_file=args.dummy_file)
        ear.boot()
    except Exception as e:
        ProjectLogger().error(f'Fatal error occurred : {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s local ear')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--device-idx', type=int, default=-1, help='Microphone identifier')
    parser.add_argument('--target-url', type=str, default='localhost:9999', help='Brain target URL')
    parser.add_argument('--dummy-file', type=str, help='Play file instead of Brain\'s responses')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
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
