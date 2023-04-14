from typing import Generator
from hyperion.utils.logger import ProjectLogger

import cv2


class Webcam:

    def __init__(self, device_idx=0, width=640, height=480):
        self.width = width
        self.height = height
        self.running = False

        self._device = device_idx
        self._generator: Generator = ...

    def __enter__(self):
        self._cap = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.framerate = self._cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *args):
        self.close()

    def __call__(self, *args, **kwargs):
        self.running = True
        self._generator = self._init_generator()
        return self._generator

    def close(self):
        self.running = False

    def _init_generator(self):
        while self._cap.isOpened() and self.running:
            ret, frame = self._cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                continue
            yield frame
        ProjectLogger().info('Camera stream stopped.')


if __name__ == '__main__':
    import requests

    with Webcam() as webcam:
        print(int(webcam.framerate))
        stream = webcam()
        idx = 0
        for frame in stream:
            cv2.imshow('OpenCV2 Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if idx == 0:
                payload = [
                    ('frame', ('frame', frame, 'image/jpeg')),
                ]
                res = requests.post(url=f'http://deepbox:9999/video', files=payload, stream=True)

            idx += 1
            idx = idx % (5 * int(webcam.framerate))
    cv2.destroyAllWindows()
