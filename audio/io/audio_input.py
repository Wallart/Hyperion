from utils.threading import Producer
from utils.logger import ProjectLogger


class AudioInput(Producer):

    def __init__(self, source):
        super().__init__()
        self._source = source

    def stop(self):
        super().stop()
        self._source.close()

    def run(self) -> None:
        self._source.open()
        generator = self._source()
        for audio_chunk in generator:
            # if not self.running:
            #     break
            self._dispatch(audio_chunk)

        ProjectLogger().info('Audio input closed.')
