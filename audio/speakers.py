from utils.threading import Consumer

import sounddevice as sd


class SpeakersStream(Consumer):

    def __init__(self, sample_rate, channels):
        super().__init__()
        self._speakers = sd.OutputStream(sample_rate, channels=channels, dtype=np.float32)

    # def __del__(self):
    #     self._speakers.stop()
    #     self._speakers.close()

    def run(self) -> None:
        self._speakers.start()
        while True:
            audio = self._in_queue.get()
            self._speakers.write(audio)
