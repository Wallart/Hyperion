from uuid import uuid4
from pyaudio import PyAudio, paFloat32

import os
import argparse
import numpy as np
import soundfile as sf


INPRINT_DURATION = 5
SAMPLING_RATE = 16000
CHUNK_DURATION_MS = 512
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION_MS / 1000)


def list_devices():
    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    microphones = {}
    for i in range(0, num_devices):
        device = audio.get_device_info_by_host_api_device_index(0, i)
        device_name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
        is_microphone = device.get('maxInputChannels') > 0
        if is_microphone:
            microphones[i] = device_name

    return microphones


def prompt_device_idx():
    while True:
        devices_dict = list_devices()
        print(devices_dict)
        device_idx = int(input('Please select device :'))
        if device_idx in devices_dict:
            break

    return device_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record voice inprint for a new speaker')
    parser.add_argument('name', type=str, help='Speaker name.')
    args = parser.parse_args()

    audio = PyAudio()
    device_idx = prompt_device_idx()
    opts = {
        'channels': 1,
        'input': True,
        'start': False,
        'format': paFloat32,
        'rate': SAMPLING_RATE,
        'frames_per_buffer': CHUNK_SIZE,
        'input_device_index': device_idx
    }

    stream = audio.open(**opts)
    stream.start_stream()

    buffer = None
    recording_size = INPRINT_DURATION * SAMPLING_RATE
    while buffer is None or len(buffer) < recording_size:
        raw_buffer = stream.read(CHUNK_SIZE, exception_on_overflow=True)
        chunk = np.frombuffer(raw_buffer, dtype=np.float32)
        buffer = chunk if buffer is None else np.concatenate([buffer, chunk], axis=0)

    wav_signal = buffer[:recording_size, ...]

    stream.stop_stream()
    stream.close()
    audio.terminate()

    outdir = os.path.join(os.getcwd(), 'resources', 'speakers_samples', args.name)
    os.makedirs(outdir, exist_ok=True)
    sf.write(os.path.join(outdir, f'{uuid4()}.wav'), wav_signal, SAMPLING_RATE)
