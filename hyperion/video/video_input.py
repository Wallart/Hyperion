from hyperion.video.io.webcam import Webcam
from hyperion.utils.threading import Producer
from hyperion.utils.logger import ProjectLogger

import requests


class VideoInput(Producer):

    def __init__(self, target, interval=3):
        super().__init__()
        self._target = target
        self._interval = interval
        self.width = 640
        self.height = 480
        self.webcam = None

    def stop(self):
        super().stop()
        if self.webcam is not None:
            self.webcam.close()

    def run(self) -> None:
        with Webcam(width=self.width, height=self.height) as self.webcam:
            idx = 0
            stream = self.webcam()
            ProjectLogger().info(f'Video stream at {self.webcam.width}x{self.webcam.height}@{self.webcam.framerate} opened.')
            while self.running:
                for frame in stream:
                    self._dispatch(frame)
                    if idx == 0:
                        opts = {
                            'url': f'{self._target}/video',
                            'stream': True,
                            'files': [
                                ('frame', ('frame', frame, 'image/jpeg')),
                            ],
                            'headers': {
                                'frame_width': str(self.webcam.width),
                                'frame_height': str(self.webcam.height),
                                'frame_channels': '3'
                            }
                        }
                        res = requests.post(**opts)
                        if res.status_code != 200:
                            ProjectLogger().error('Server error occurred during video stream')
                            break

                    idx += 1
                    idx = idx % (self._interval * int(self.webcam.framerate))
                self._dispatch(None)
        ProjectLogger().info('Camera closed.')
