from time import time
from daemonocle import Daemon
from audio.io.audio_input import AudioInput
from audio.io.source.in_file import InFile
from audio.io.source.in_device import InDevice
from audio.io.audio_output import AudioOutput
from utils.utils import get_ctx, frame_decode
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

    def __init__(self, ctx, target_url, in_idx, out_idx, dummy_file=None):
        self._ctx = ctx
        self._in_sample_rate = 16000
        self._out_sample_rate = 24000
        self._target_url = f'http://{target_url}'
        self._dummy_file = dummy_file if dummy_file is None else os.path.expanduser(dummy_file)

        source = InDevice(in_idx, self._in_sample_rate) if self._dummy_file is None else InFile(self._dummy_file, self._in_sample_rate)
        audio_in = AudioInput(source)
        detector = VoiceDetector(ctx, self._in_sample_rate, activation_threshold=.9)
        recognizer = VoiceRecognizer(ctx)
        audio_out = AudioOutput(out_idx, self._out_sample_rate)

        self.sink = audio_in.pipe(detector).pipe(recognizer).create_sink()
        self.intake = audio_out.create_intake()
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
        ProjectLogger().info('Processing Julien\'s request...')
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
                audio_chunk, recognized_speaker = self.sink.get()
                audio_chunk = audio_chunk.numpy()
                if recognized_speaker == 'Unknown':
                    ProjectLogger().info('Request ignored.')
                    continue

                _ = executor.submit(self._process_request, recognized_speaker, audio_chunk)


APP_NAME = 'hyperion_local_ear'


def main():
    try:
        ctx = get_ctx(args)
        ear = LocalEar(ctx, args.target_url, args.in_idx, args.out_idx, dummy_file=args.dummy_file)
        ear.boot()
    except Exception as e:
        ProjectLogger().error(f'Fatal error occurred : {e}')
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s local ear')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--in-idx', type=int, default=-1, help='Audio input identifier')
    parser.add_argument('--out-idx', type=int, default=-1, help='Audio output identifier')
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
