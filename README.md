# Hyperion 

ChatGPT based vocal assistant.

## Description

Hyperion offers natural vocal interation with OpenAI's GPT models.
The followings features are supported :
- Vocal recognition
- Speech-to-text
- Text-to-Speech
- Keyword spotting

**Supported platforms:**

- [X] macOS
- [X] Linux
- [X] Windows

---
## Usage


**Install dependencies. (Python 3.10 is required)**
```bash
# Common dependencies
pip install -r requirements.txt
# Client side dependencies (macOS / Linux only)
sudo apt install libportaudio2 libsndfile1-dev python3-tk
# Server side dependency
sudo apt install ffmpeg
```

**Start applications**
```bash
# Start server
python3 hyperion_server.py start
# Start client
python3 hyperion_client.py --target-url <SERVER_URL>:<SERVER_PORT>
```

## Docker
The server can also be built and started as a docker image
```bash
# Start image build process from project's root directory
docker build -t hyperion_server .
# Start server
docker run --restart=always --gpus all -d --name hyperion -p 6450:6450 -v ~/.hyperion/whisper:/root/.hyperion/whisper -v ~/.cache:/root/.cache hyperion_server
```

## Known issues
**Server responses are not displayed in GUI**
- Client and/or server OS clock are not synchronized : Sync system clock with an NTP server 
