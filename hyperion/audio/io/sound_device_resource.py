from abc import ABC
from time import sleep
from hyperion.utils.logger import ProjectLogger

import sounddevice as sd


class SoundDeviceResource(ABC):

    def __init__(self, device_idx, output, sample_rate, duration_ms=512, channels=1):
        self.channels = channels
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.chunk_size = int(sample_rate * self.duration_ms / 1000)
        self.is_closing = False  # Marked for closing
        self.output = output

        self.device_type = 'Output' if self.output else 'Input'
        self.default_device = sd.default.device[1] if self.output else sd.default.device[0]
        self.device_idx = None
        self.device_name = None
        try:
            device_idx = self.default_device if device_idx == -1 else device_idx
            device = sd.query_devices(device_idx, kind=self.device_type.lower())
            self.device_idx = device['index']
            self.device_name = device['name']
        except Exception as e:
            self._prompt_device_idx()

        self._init_stream()

    def _init_stream(self):
        # important to avoid crackling sound when playing sound on speakers
        # https://macreports.com/how-to-fix-the-popping-and-crackling-sound-on-mac/#:~:text=This%20popping%20or%20crackling%20sound,the%20format%20can%20fix%20this.
        self.device_default_sr = sd.query_devices(self.device_idx)['default_samplerate']
        opts = {
            'device': self.device_idx,
            'channels': self.channels,
            # 'samplerate': sample_rate
        }
        if self.output:
            self._stream = sd.OutputStream(**opts, samplerate=self.device_default_sr)
        else:
            self._stream = sd.InputStream(**opts, blocksize=self.chunk_size, samplerate=self.sample_rate)

    # def get_callback(self):
    #     return None

    def open(self):
        self._stream.start()
        ProjectLogger().info(f'{self.device_type} device {self.device_name} opened.')

    def close(self):
        self.is_closing = True
        sleep(0.5)  # let some time for the listening loop to stop

        if self._stream.active:
            self._stream.stop()
        self._stream.close()
        ProjectLogger().info(f'{self.device_type} device {self.device_name} closed.')

    @staticmethod
    def list_devices(device_type):
        prop = 'max_output_channels' if device_type == 'Output' else 'max_input_channels'
        valid_devices = {}
        for device in sd.query_devices():
            if device[prop] > 0:
                valid_devices[device['index']] = device['name']

        return valid_devices

    def _prompt_device_idx(self):
        while True:
            devices_dict = SoundDeviceResource.list_devices(self.device_type)
            print(devices_dict)
            self.device_idx = input(f'Select {self.device_type.lower()} device :')
            try:
                self.device_idx = int(self.device_idx)
                if self.device_idx in devices_dict:
                    self.device_name = devices_dict[self.device_idx]
                    break
            except ValueError:
                continue
