from hyperion.video.io import VideoDevices
from hyperion.video.io.webcam import Webcam
from hyperion.utils.threading import Producer
from hyperion.utils.logger import ProjectLogger

import requests


class VideoInput(Producer):

    def __init__(self, device, target, interval=3):
        super().__init__()
        self._target = target
        self._interval = interval

        self.device = device
        self.device_idx = VideoDevices.query_device(device)['index']

        self.width = 640
        self.height = 480
        self.webcam = None

    def stop(self):
        super().stop()
        if self.webcam is not None:
            self.webcam.close()

    def run(self) -> None:
        with Webcam(device_idx=self.device_idx, width=self.width, height=self.height) as self.webcam:
            idx = 0
            stream = self.webcam()
            specs = f'{self.webcam.width}x{self.webcam.height}@{self.webcam.framerate}'
            ProjectLogger().info(f'Video stream {specs} on device {self.device_idx} opened.')
            while self.running:
                for frame in stream:
                    height, width, channels = frame.shape
                    self._dispatch(frame)
                    if idx == 0:
                        opts = {
                            'url': f'{self._target}/video',
                            'stream': True,
                            'files': [
                                ('frame', ('frame', frame, 'image/jpeg')),
                            ],
                            'headers': {
                                'frame_width': str(width),
                                'frame_height': str(height),
                                'frame_channels': str(channels)
                            }
                        }
                        res = requests.post(**opts)
                        if res.status_code != 200:
                            ProjectLogger().error('Server error occurred during video stream')
                            break

                    idx += 1
                    idx = idx % (self._interval * int(self.webcam.framerate))
                self._dispatch(None)
        ProjectLogger().info(f'Camera {self.device_idx} closed.')
