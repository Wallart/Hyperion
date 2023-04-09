from abc import ABC
from utils.logger import ProjectLogger

import sounddevice as sd


class SoundDeviceResource(ABC):

    def __init__(self, device_idx, output, sample_rate, duration_ms=512, channels=1):
        self.channels = channels
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.chunk_size = int(sample_rate * self.duration_ms / 1000)

        self.device_type = 'Output' if output else 'Input'
        self.device_idx = None
        self.device_name = None
        try:
            device = sd.query_devices(device_idx, kind=self.device_type.lower())
            self.device_idx = device['index']
            self.device_name = device['name']
        except Exception as e:
            self._prompt_device_idx()

        opts = {
            'device': self.device_idx,
            'channels': channels,
            'samplerate': sample_rate
        }
        if output:
            self._stream = sd.OutputStream(**opts)
        else:
            self._stream = sd.InputStream(**opts, blocksize=self.chunk_size)

    def get_callback(self):
        return None

    def open(self):
        self._stream.start()
        ProjectLogger().info(f'{self.device_type} device {self.device_name} opened.')

    def close(self):
        if self._stream.active:
            self._stream.stop()
        ProjectLogger().info(f'{self.device_type} device {self.device_name} closed.')

    def list_devices(self):
        prop = 'max_output_channels' if self.device_type == 'Output' else 'max_input_channels'
        valid_devices = {}
        for device in sd.query_devices():
            if device[prop] > 0:
                valid_devices[device['index']] = device['name']

        return valid_devices

    def _prompt_device_idx(self):
        while True:
            devices_dict = self.list_devices()
            print(devices_dict)
            self.device_idx = input(f'Select {self.device_type.lower()} device :')
            try:
                self.device_idx = int(self.device_idx)
                if self.device_idx in devices_dict:
                    self.device_name = devices_dict[self.device_idx]
                    break
            except ValueError:
                continue
