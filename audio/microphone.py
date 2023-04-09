# from queue import Queue
from librosa import resample
from pyaudio import PyAudio, paInt16
# from audio.aec.time_domain_adaptive_filters.apa import apa
# from audio.aec.time_domain_adaptive_filters.rls import rls
from audio.audio_source import AudioSource
from audio import int16_to_float32, find_offset, float64_to_int16

import audioop
import logging
import numpy as np
import noisereduce as nr


class Microphone(AudioSource):

    def __init__(self, device_idx=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_idx = -1
        self._pyaudio = PyAudio()
        self._listening = False

        # self._current_feedback = None
        # self._feedback_queue = Queue()

        if device_idx > -1:
            devices_dict = self.list_devices()
            assert device_idx in devices_dict, 'Microphone not found.'
            self._device_idx = device_idx
        else:
            self._prompt_device_idx()

        opts = {
            'input': True,
            'start': False,
            'format': paInt16,
            'channels': self._channels,
            'rate': self._sampling_rate,
            'frames_per_buffer': self._chunk_size,
            'input_device_index': self._device_idx
        }
        self._stream = self._pyaudio.open(**opts)

    # def set_feedback(self, feedback, sample_rate):
    #     if sample_rate != self._sampling_rate:
    #         feedback = resample(int16_to_float32(feedback), orig_sr=sample_rate, target_sr=self._sampling_rate)
    #     self._feedback_queue.put(feedback)

    # def _consume_feedback_chunk(self, searched_chunk):
    #     if self._current_feedback is None and self._feedback_queue.empty():
    #         return None
    #
    #     if not self._feedback_queue.empty():
    #         extract = self._feedback_queue.get()
    #         self._current_feedback = extract if self._current_feedback is None else np.concatenate([self._current_feedback, extract])
    #
    #     if len(self._current_feedback) > self._chunk_size:
    #         offset = find_offset(self._current_feedback, searched_chunk, self._sampling_rate)
    #         if offset == 0:
    #             return None
    #
    #         found_chunk = self._current_feedback[offset:offset+self._chunk_size]
    #         self._current_feedback = self._current_feedback[offset + self._chunk_size:]
    #     else:
    #         found_chunk = np.pad(self._current_feedback, (0, self._chunk_size - len(self._current_feedback)))
    #         self._current_feedback = None
    #
    #     return found_chunk

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

    # def acoustic_echo_cancellation(self, chunk, feedback_chunk):
    #     aec_ed = apa(feedback_chunk, chunk, N=256, P=5, mu=0.1)
    #     aec_ed = np.clip(aec_ed, -1, 1)
    #     aec_ed = float64_to_int16(aec_ed)
    #     return aec_ed

    def _init_generator(self):
        listening = False
        listened_chunks = 0
        while True:
            raw_buffer = self._stream.read(self._chunk_size, exception_on_overflow=False)
            buffer = nr.reduce_noise(np.frombuffer(raw_buffer, dtype=np.int16), self._sampling_rate)

            rms = audioop.rms(buffer, 2)  # root mean square of signal to detect if there is interesting things to record
            if rms >= 1000:
                listening = True
            elif rms < 1000 and listened_chunks >= 4:  # eq to 2 sec of silence
                listening = False
                listened_chunks = 0

            if listening:
                listened_chunks += 1
                chunk = int16_to_float32(buffer)

                # feedback_chunk = self._consume_feedback_chunk(chunk)
                # if feedback_chunk is None:
                #     yield chunk
                # else:
                #     aec_chunk = self.acoustic_echo_cancellation(chunk, feedback_chunk)
                #     aec_chunk = nr.reduce_noise(aec_chunk, self._sampling_rate)
                #     new_rms = audioop.rms(aec_chunk, 2)
                #     logging.info(f'RMS {rms} -> {new_rms}')
                #     if new_rms < rms:
                #         yield None
                #     else:
                yield chunk
            else:
                yield None  # sending silence token

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
