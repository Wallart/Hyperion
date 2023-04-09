from time import time
from audio.io.audio_input import AudioInput
from audio.io.source.in_file import InFile
from audio.io.source.in_device import InDevice
from audio.io.audio_output import AudioOutput
from utils.execution import startup, handle_errors
from utils.utils import get_ctx
from utils.protocol import frame_decode
# from audio.audio_file import AudioFile
from utils.logger import ProjectLogger
from concurrent.futures import ThreadPoolExecutor
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import os
import argparse
import requests
import numpy as np


class LocalEar:

    def __init__(self, ctx, opts):
        self._ctx = ctx
        self._in_sample_rate = 16000
        self._out_sample_rate = 24000
        self._target_url = f'http://{opts.target_url}'
        self._dummy_file = opts.dummy_file if opts.dummy_file is None else os.path.expanduser(opts.dummy_file)
        self._no_recog = opts.no_recog

        source = InDevice(opts.in_idx, self._in_sample_rate, rms=opts.rms) if self._dummy_file is None else InFile(self._dummy_file, self._in_sample_rate)
        audio_in = AudioInput(source)
        detector = VoiceDetector(ctx, self._in_sample_rate, activation_threshold=.9)
        audio_out = AudioOutput(opts.out_idx, self._out_sample_rate)
        self.intake = audio_out.create_intake()

        if opts.no_recog:
            self.sink = audio_in.pipe(detector).create_sink()
            self.threads = [audio_in, detector, audio_out]
        else:
            recognizer = VoiceRecognizer(ctx)
            self.sink = audio_in.pipe(detector).pipe(recognizer).create_sink()
            self.threads = [audio_in, detector, recognizer, audio_out]

    def boot(self):
        try:
            _ = [t.start() for t in self.threads]
            self.mainloop()
        except KeyboardInterrupt as interrupt:
            _ = [t.stop() for t in self.threads]

        # _ = [t.join() for t in self.threads]
        _ = [t.join() for t in self.threads[1:]]

    def _process_request(self, recognized_speaker, audio_chunk):
        t0 = time()
        ProjectLogger().info(f'Processing {recognized_speaker}\'s request...')
        payload = [
            ('speaker', ('speaker', recognized_speaker, 'text/plain')),
            ('speech', ('speech', audio_chunk.tobytes(), 'application/octet-stream'))
        ]

        try:
            res = requests.post(url=f'{self._target_url}/audio', files=payload, stream=True)
            if res.status_code != 200:
                ProjectLogger().warning(f'Something went wrong. HTTP {res.status_code}')
                return

            buffer = bytearray()
            for bytes_chunk in res.iter_content(chunk_size=4096):
                buffer.extend(bytes_chunk)
                output = frame_decode(buffer)
                if output is None:
                    continue

                decoded_frame, buffer = output
                self.distribute(recognized_speaker, decoded_frame)

            ProjectLogger().info(f'{recognized_speaker}\'s request processed in {time() - t0:.3f} sec(s).')
        except Exception as e:
            ProjectLogger().warning(f'Request canceled : {e}')

    def distribute(self, recognized_speaker, decoded_frame):
        request = decoded_frame['REQ']
        answer = decoded_frame['ANS']
        audio = decoded_frame['PCM']

        ProjectLogger().info(f'{recognized_speaker} : {request}')
        ProjectLogger().info(f'ChatGPT : {answer}')
        spoken_chunk = np.frombuffer(audio, dtype=np.int16)
        self.intake.put(spoken_chunk)
        # self.source.set_feedback(spoken_chunk, self._out_sample_rate)

    def mainloop(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                data = self.sink.drain()
                audio_chunk, recognized_speaker = data, 'Unknown'
                if not self._no_recog:
                    audio_chunk, recognized_speaker = data
                audio_chunk = audio_chunk.numpy()
                if recognized_speaker == 'Unknown' and not self._no_recog:
                    ProjectLogger().info('Request ignored.')
                    continue

                _ = executor.submit(self._process_request, recognized_speaker, audio_chunk)


APP_NAME = 'hyperion_local_ear'


@handle_errors
def main(args):
    ctx = get_ctx(args)
    ear = LocalEar(ctx, args)
    ear.boot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s local ear')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon')
    parser.add_argument('--debug', action='store_true', help='Enables debugging.')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')

    parser.add_argument('--in-idx', type=int, default=-1, help='Audio input identifier')
    parser.add_argument('--rms', type=int, default=1000, help='Sound detection threshold')
    parser.add_argument('--out-idx', type=int, default=-1, help='Audio output identifier')
    parser.add_argument('--target-url', type=str, default='localhost:9999', help='Brain target URL')
    parser.add_argument('--dummy-file', type=str, help='Play file instead of Brain\'s responses')
    parser.add_argument('--no-recog', action='store_true', help='Start bot without user recognition.')
    
    startup(APP_NAME.lower(), parser, main)
