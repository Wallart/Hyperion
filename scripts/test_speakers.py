import os
import numpy as np
import soundfile as sf
import sounddevice as sd


if __name__ == '__main__':
    path = os.path.expanduser('~/datasets/test.wav')
    (fs, sr) = sf.read(path, dtype=np.int16)
    sd.play(fs, samplerate=sr)
    sd.wait()
