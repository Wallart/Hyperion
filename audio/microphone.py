from array import array
from pyaudio import PyAudio, paFloat32, paInt16
from audio.audio_source import AudioSource

import math
import struct
import audioop
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

        self._format = paInt16
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

    @staticmethod
    def rms(data):
        count = len(data) / 2
        format = "%dh" % (count)
        shorts = struct.unpack(format, data)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0 / 32768)
            sum_squares += n * n
        return math.sqrt(sum_squares / count)

    @staticmethod
    def np_audioop_rms(data, width):
        """audioop.rms() using numpy; avoids another dependency for app"""
        # _checkParameters(data, width)
        if len(data) == 0: return None
        fromType = (np.int8, np.int16, np.int32)[width // 2]
        d = np.frombuffer(data, fromType).astype(np.float)
        rms = np.sqrt(np.mean(d ** 2))
        return int(rms)

    def _init_generator(self):
        listening = False
        listened_chunks = 0
        while True:
            raw_buffer = self._stream.read(self._chunk_size, exception_on_overflow=False)

            rms = audioop.rms(raw_buffer, 2)  # root mean square of signal to detect if there is interesting things to record
            if rms >= 1000:
                listening = True
            elif rms < 1000 and listened_chunks >= 4:  # eq to 2 sec of silence
                listening = False
                listened_chunks = 0

            if listening:
                listened_chunks += 1
                chunk = np.frombuffer(raw_buffer, dtype=np.int16)
                chunk = chunk.astype(np.float32) / (((2 ** 16) / 2) - 1)
                yield chunk
            else:
                yield None

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
