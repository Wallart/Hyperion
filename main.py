from queue import Queue
from functools import partial

from speechbrain.dataio.preprocess import AudioNormalizer

from utils.utils import LivePlotter, SpeakersStream
from audio.audio_file import AudioFile
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_transcriber import VoiceTranscriber
from time import sleep

import torch
import logging
import numpy as np


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # audio_clazz = partial(AudioFile, '~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    # audio_clazz = partial(AudioFile, '~/Desktop/test.wav')
    audio_clazz = partial(Microphone, device_idx=0)

    with audio_clazz(duration_ms=512) as source:
        # Threaded operations
        input_stream = AudioStream(source)
        output_stream = SpeakersStream(source.sample_rate(), source.channels())
        detector = VoiceDetector(source.sample_rate(), activation_threshold=.7)
        transcriber = VoiceTranscriber()

        # other
        plotter = LivePlotter(source.name(), source.chunk_duration(), source.chunk_size())

        input_stream.register(detector)
        input_stream.register(output_stream)
        detector.register(transcriber)
        # normalizer = AudioNormalizer()

        try:
            input_stream.start()
            output_stream.start()
            detector.start()
            transcriber.start()

            q1 = input_stream.create_queue()
            sleep(5)
            logging.info('STARTED !')

            while True:
                plotter.draw(q1.get(), None, None)

        except KeyboardInterrupt as e:
            logging.error('Interruption received.')
