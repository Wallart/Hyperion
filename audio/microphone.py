from array import array
from pyaudio import PyAudio, paFloat32
from audio.audio_source import AudioSource

import logging
import numpy as np


class Microphone(AudioSource):

    def __init__(self, device_idx=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_idx = -1
        self._pyaudio = PyAudio()
        self._listening = False

        if device_idx > -1:
            devices_dict = self.list_devices()
            assert device_idx in devices_dict, 'Microphone not found.'
            self._device_idx = device_idx
        else:
            self._prompt_device_idx()

        self._format = paFloat32
        # self._padding_duration_ms = 1500  # 1 sec jugement
        # self._num_padding_chunks = int(self._padding_duration_ms / self._chunk_duration_ms)
        # self._num_window_chunks = int(400 / self._chunk_duration_ms)  # 400 ms/ 30ms  ge
        # self._num_window_chunks_end = self._num_window_chunks * 2
        opts = {
            'input': True,
            'start': False,
            'format': self._format,
            'channels': self._channels,
            'rate': self._sampling_rate,
            'frames_per_buffer': self._chunk_size,
            'input_device_index': self._device_idx
        }
        self._stream = self._pyaudio.open(**opts)

    def start(self):
        logging.info(f'Start listening device {self.name()}')
        self._stream.start_stream()
        self._listening = True
        self._generator = self._init_generator()

    def close(self):
        self._listening = False
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()
        logging.info(f'Closed device {self.name()}')

    @staticmethod
    def normalize(chunk, maximum=32767):
        """
        "Average the volume out - # 16384"
        :param chunk:
        :param maximum:
        :return:
        """
        times = float(maximum) / max(abs(i) for i in chunk)
        r = array('h')
        for i in chunk:
            r.append(int(i * times))
        return r

    def _init_generator(self):
        while True:
            raw_buffer = self._stream.read(self._chunk_size, exception_on_overflow=False)
            chunk = np.frombuffer(raw_buffer, dtype=np.float32)
            # raw_data = array('h')
            # raw_data.extend(array('h', buffer))
            # raw_data = AudioEngine.normalize(raw_data)
            yield chunk

    def name(self):
        return self.list_devices()[self._device_idx]

    def list_devices(self):
        info = self._pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        microphones = {}
        for i in range(0, num_devices):
            device = self._pyaudio.get_device_info_by_host_api_device_index(0, i)
            device_name = self._pyaudio.get_device_info_by_host_api_device_index(0, i).get('name')
            is_microphone = device.get('maxInputChannels') > 0
            if is_microphone:
                microphones[i] = device_name

        return microphones

    def _prompt_device_idx(self):
        while True:
            devices_dict = self.list_devices()
            print(devices_dict)
            self._device_idx = int(input('Please select device :'))
            if self._device_idx in devices_dict:
                break

        return self._device_idx
