from abc import ABC
from pyaudio import PyAudio, paInt16
from utils.logger import ProjectLogger


class PyAudioResource(ABC):

    def __init__(self, device_idx, output, sample_rate, channels=1):
        self.opened = False
        self.channels = channels
        self.sample_rate = sample_rate

        self._output = output
        self._pyaudio = PyAudio()
        if device_idx > -1:
            devices_dict = self.list_devices()
            device_type = 'Output' if output else 'Input'
            self._device_idx = device_idx
            if device_idx not in devices_dict:
                ProjectLogger().warning(f'{device_type} device {device_idx} not found.')
                self._prompt_device_idx()
        else:
            self._prompt_device_idx()

        opts = {
            'start': False,
            'input': not output,
            'output': output,
            'format': paInt16,
            'channels': channels,
            'rate': sample_rate
        }
        self._stream = self._pyaudio.open(**opts)

    def open(self):
        self.opened = True
        self._stream.start_stream()
        device_type = 'Output' if self._output else 'Input'
        ProjectLogger().info(f'{device_type} device {self.device_name()} opened.')

    def close(self):
        self.opened = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

        device_type = 'Output' if self._output else 'Input'
        ProjectLogger().info(f'{device_type} device {self.device_name()} closed.')
        self._pyaudio.terminate()

    def list_devices(self):
        info = self._pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        valid_devices = {}
        prop = 'maxOutputChannels' if self._output else 'maxInputChannels'
        for i in range(0, num_devices):
            device = self._pyaudio.get_device_info_by_host_api_device_index(0, i)
            device_name = self._pyaudio.get_device_info_by_host_api_device_index(0, i).get('name')
            is_valid = device.get(prop) > 0
            if is_valid:
                valid_devices[i] = device_name

        return valid_devices

    def _prompt_device_idx(self):
        while True:
            devices_dict = self.list_devices()
            print(devices_dict)
            self._device_idx = int(input('Please select device :'))
            if self._device_idx in devices_dict:
                break

        return self._device_idx

    def device_name(self):
        return self.list_devices()[self._device_idx]
