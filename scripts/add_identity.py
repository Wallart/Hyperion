from uuid import uuid4
from hyperion.utils.logger import ProjectLogger
from hyperion.audio.io.source.in_device import InDevice

import os
import argparse
import numpy as np
import soundfile as sf


INPRINT_DURATION = 5
SAMPLING_RATE = 16000
CHUNK_DURATION_MS = 512
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION_MS / 1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record voice inprint for a new speaker')
    parser.add_argument('name', type=str, help='Speaker name.')
    parser.add_argument('samples_dir', default='~/.hyperion/speakers_samples', type=str, help='Speaker samples directory.')
    args = parser.parse_args()

    args.samples_dir = os.path.expanduser(args.samples_dir)

    mic = InDevice(-2, 16000, db=60)
    mic.open()

    buffer = None
    recording_size = INPRINT_DURATION * SAMPLING_RATE
    stream = mic()
    for chunk in stream:
        ProjectLogger().info('Sound chunk received.')
        if buffer is not None and len(buffer) >= recording_size:
            break
        if chunk is None:
            continue
        buffer = chunk if buffer is None else np.concatenate([buffer, chunk], axis=0)

    mic.close()

    outdir = os.path.join(args.samples_dir, args.name)
    os.makedirs(outdir, exist_ok=True)
    sf.write(os.path.join(outdir, f'{uuid4()}.wav'), buffer[:recording_size, ...], SAMPLING_RATE)
