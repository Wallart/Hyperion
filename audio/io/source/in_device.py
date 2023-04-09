from queue import Queue
from librosa import resample
from functools import partial
from utils.logger import ProjectLogger
from audio.io.source import AudioSource
from audio.aec.nonlinear_adaptive_filters import *
from audio.aec.time_domain_adaptive_filters import *
from audio.aec.frequency_domain_adaptive_filters import *
from audio.io.sound_device_resource import SoundDeviceResource
from audio import float32_to_int16, float64_to_int16, find_offset, int16_to_float32

import audioop
import numpy as np
import noisereduce as nr


class InDevice(SoundDeviceResource, AudioSource):

    def __init__(self, device_idx, sample_rate, rms=1000, **kwargs):
        super().__init__(device_idx, False, sample_rate, **kwargs)
        self.rms_threshold = rms
        self._prev_buffer = np.zeros((self.chunk_size,), dtype=np.float32)
        self._current_feedback = None
        self._feedback_queue = Queue()

    def set_feedback(self, feedback, sample_rate):
        if sample_rate != self.sample_rate:
            feedback = resample(int16_to_float32(feedback), orig_sr=sample_rate, target_sr=self.sample_rate)
        self._feedback_queue.put(feedback)

    def _consume_feedback_chunk(self, searched_chunk):
        if self._current_feedback is None and self._feedback_queue.empty():
            return None

        if not self._feedback_queue.empty():
            extract = self._feedback_queue.get()
            self._feedback_queue.task_done()
            self._current_feedback = extract if self._current_feedback is None else np.concatenate([self._current_feedback, extract])

        if len(self._current_feedback) > self.chunk_size:
            offset = find_offset(self._current_feedback, searched_chunk, self.sample_rate)
            if offset == 0:
                return None

            found_chunk = self._current_feedback[offset:offset+self.chunk_size]
            self._current_feedback = self._current_feedback[offset + self.chunk_size:]
        else:
            found_chunk = np.pad(self._current_feedback, (0, self.chunk_size - len(self._current_feedback)))
            self._current_feedback = None

        return found_chunk

    @staticmethod
    def acoustic_echo_cancellation(chunk, feedback_chunk, algorithm=partial(apa, N=256, P=5, mu=0.1)):
        aec_ed = algorithm(feedback_chunk, chunk)
        aec_ed = np.clip(aec_ed, -1, 1)
        aec_ed = float64_to_int16(aec_ed)
        return aec_ed

    def _init_generator(self):
        listening = False
        listened_chunks = 0
        while self._stream.active and not self.is_closing:
            buffer, overflowed = self._stream.read(self.chunk_size)
            buffer = buffer.squeeze()
            if buffer.sum() == 0:
                continue
            buffer = nr.reduce_noise(buffer, self.sample_rate)

            # root mean square of signal to detect if there is interesting things to record
            rms = audioop.rms(float32_to_int16(buffer), 2)
            if rms >= self.rms_threshold:
                listening = True
            elif rms < self.rms_threshold and listened_chunks >= 4:  # eq to 2 sec of silence
                listening = False
                listened_chunks = 0
                ProjectLogger().info('Candidate noise detected.')

            if listening:
                listened_chunks += 1
                if listened_chunks == 1:
                    yield np.concatenate([self._prev_buffer, buffer], axis=0)
                else:
                    yield buffer

                # feedback_chunk = self._consume_feedback_chunk(buffer)
                # if feedback_chunk is None:
                #     yield buffer
                # else:
                #     aec_chunk = self.acoustic_echo_cancellation(buffer, feedback_chunk)
                #     aec_chunk = nr.reduce_noise(aec_chunk, self.sample_rate)
                #     new_rms = audioop.rms(aec_chunk, 2)
                #     ProjectLogger().info(f'Echo cancelled.')
                #     if new_rms < rms:
                #         yield None
                #     else:
                #         print(rms)
                #         yield buffer
            else:
                yield None  # sending silence token

            self._prev_buffer[:] = buffer

        ProjectLogger().info('Input device listening stopped.')
