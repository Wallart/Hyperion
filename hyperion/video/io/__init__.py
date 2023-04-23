from hyperion.utils import Singleton

import cv2
import threading


class VideoDevices(metaclass=Singleton):
    def __init__(self):
        self.video_devices_idx = None

    def list_devices(self):
        if self.video_devices_idx is None:
            self.video_devices_idx = VideoDevices.list_cameras()
        return [f'Camera {c}' for c in self.video_devices_idx]

    @staticmethod
    def query_device(name):
        index = name.split(' ')[1]
        # Other info could be fetched in the future
        device = dict(name=name, index=int(index))
        return device

    @staticmethod
    def list_cameras():
        idx = 0
        cam_indexes = []
        while True:
            cap = cv2.VideoCapture(idx)
            try:
                # windows, macOS support
                if cap.getBackendName() in ['DSHOW', 'MSMF', 'AVFOUNDATION']:
                    cam_indexes.append(idx)
            except:
                break
            finally:
                cap.release()
            idx += 1
        return cam_indexes

thread = threading.Thread(target=VideoDevices().list_devices, daemon=True)
thread.start()
