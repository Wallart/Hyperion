from utils.threading import Producer


class AudioInput(Producer):

    def __init__(self, source):
        super().__init__()
        self._source = source
        self.daemon = True

    def stop(self):
        super().stop()
        self._source.close()

    def run(self) -> None:
        self._source.open()
        generator = self._source()
        for audio_chunk in generator:
            if not self.running:
                break
            self._dispatch(audio_chunk)
