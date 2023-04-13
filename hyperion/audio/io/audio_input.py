from time import sleep
from hyperion.utils.threading import Producer
from hyperion.utils.logger import ProjectLogger


class AudioInput(Producer):

    def __init__(self, source):
        super().__init__()
        self._source = source

    def change(self, new_source):
        ProjectLogger().info('Changing Input device.')
        self._source.close()
        self._source = new_source

    def stop(self):
        super().stop()
        self._source.close()

    def run(self) -> None:
        while self.running:
            self._source.open()
            generator = self._source()
            for audio_chunk in generator:
                # if not self.running:
                #     break
                self._dispatch(audio_chunk)

            ProjectLogger().info('Audio input closed.')
            sleep(.5)
