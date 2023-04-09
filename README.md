AUDIO TO TEXT ->
Please git lfs this repo
https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large/tree/main

TTS -> FAIRSEQ Install

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq


brew install espeak