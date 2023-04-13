from typing import Generator

import cv2


class Webcam:

    def __init__(self, device_idx=0, width=640, height=480):
        self.width = width
        self.height = height

        self._device = device_idx
        self._generator: Generator = ...

    def __enter__(self):
        self._cap = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.framerate = self._cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *args):
        self._cap.release()

    def __call__(self, *args, **kwargs):
        self._generator = self._init_generator()
        return self._generator

    def _init_generator(self):
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                continue
            yield frame
