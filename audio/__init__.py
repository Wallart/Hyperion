from scipy import signal

import math
import numpy as np


def find_offset(source_signal, target_signal, sr_within, window=10):
    """
    Look for a sound within another sound
    :param source_signal:
    :param target_signal:
    :param sr_within:
    :param window:
    :return:
    """
    assert np.issubdtype(source_signal.dtype, np.float32)
    assert np.issubdtype(target_signal.dtype, np.float32)
    c = signal.correlate(source_signal, target_signal[:sr_within*window], mode='valid', method='fft')
    peak = np.argmax(c)
    print(f'max {c.max()}')
    return peak


def int16_to_float32(audio):
    assert np.issubdtype(audio.dtype, np.int16), f'Cannot convert {audio.dtype} to Float32'
    return audio.astype(np.float32) / 32767.


def float32_to_int16(audio):
    assert np.issubdtype(audio.dtype, np.float32), f'Cannot convert {audio.dtype} to Int16'
    return (audio * 32767.).astype(np.int16)


def float64_to_int16(audio):
    assert np.issubdtype(audio.dtype, np.float64), f'Cannot convert {audio.dtype} to Int16'
    return float32_to_int16(audio.astype(np.float32))


def rms_to_db(rms):
    return 20 * math.log10(rms)
