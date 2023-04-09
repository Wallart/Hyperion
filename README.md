# Hyperion 

ChatGPT based vocal assistant.

## Description

Hyperion offers natural vocal interation with OpenAI's GPT models.
The followings features are supported :
- Vocal recognition
- Speech-to-text
- Text-to-Speech
- Keyword spotting

## Usage

### Install dependencies. (Python 3.10 required)
sudo apt install libportaudio2 libsndfile1-dev python3-tk

pip install -r requirements.txt
### Start the server
python3 hyperion_server.py start
### Start the client
python3 hyperion_client.py --target-url <SERVER_URL>