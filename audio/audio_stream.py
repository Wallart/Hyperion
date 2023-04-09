from utils.threading import Producer
from audio.audio_source import AudioSource


class AudioStream(Producer):
    def __init__(self, audio_source: AudioSource):
        super().__init__()
        self._audio_source = audio_source

    def __del__(self):
        self.close()

    def close(self):
        self._audio_source.close()

    def run(self) -> None:
        _ = [self._dispatch(chunk) for chunk in self._audio_source.read()]

