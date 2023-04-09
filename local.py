from functools import partial
from brain.chat_gpt import ChatGPT
from utils.utils import LivePlotter, TypeWriter
from audio.speakers_stream import SpeakersStream
from audio.microphone import Microphone
from audio.audio_stream import AudioStream
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber
from time import sleep
from queue import Queue

import logging


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # audio_clazz = partial(AudioFile, '~/Desktop/709c2426101b93ce09d033eac48a56efe1a79e99.wav')
    # audio_clazz = partial(AudioFile, '~/datasets/test.wav')
    audio_clazz = partial(Microphone, device_idx=0)

    with audio_clazz(duration_ms=512) as source:
        # Threaded operations
        input_stream = AudioStream(source)
        # output_stream = SpeakersStream(source.sample_rate(), source.channels())
        detector = VoiceDetector(source.sample_rate(), activation_threshold=.9)
        transcriber = VoiceTranscriber()
        synthesizer = VoiceSynthesizer()
        output_stream = SpeakersStream(synthesizer.sample_rate, source.channels())
        writer = TypeWriter()

        # other
        chat = ChatGPT()
        plotter = LivePlotter(source.name(), source.chunk_duration(), source.chunk_size())

        # Pipeline part 1
        input_stream.pipe(detector).pipe(transcriber)
        # Pipeline part 2
        synthesizer.pipe(output_stream)

        try:
            input_stream.start()
            output_stream.start()
            detector.start()
            transcriber.start()
            # synthesizer.start()

            # q1 = input_stream.create_queue()
            q2 = transcriber.create_sink()
            q3 = transcriber.create_sink()
            q4 = synthesizer.create_intake()
            writer.set_in_queue(q3)

            synthesizer.start()
            writer.start()

            sleep(5)
            logging.info('STARTED !')

            while True:
                # plotter.draw(q1.get(), None, None)
                response = chat.answer(q2.get())
                print(f'ChatGPT : {response}')
                q4.put(response)

        except KeyboardInterrupt as e:
            logging.error('Interruption received.')
