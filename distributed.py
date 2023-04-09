from brain.chat_gpt import ChatGPT
from flask import Flask, Response, request
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber
from requests_toolbelt import MultipartEncoder

import logging

import numpy as np

app = Flask(__name__)


@app.route('/audio', methods=['POST'])
def audio_stream():
    speech = request.files['speech'].read()
    speaker = request.files['speaker'].read().decode('utf-8')

    audio_chunk = np.frombuffer(speech, dtype=np.float32)
    intake_1.put(audio_chunk)
    transcription = sink_1.get()
    if transcription is None:
        return Response(response='Speak louder motherfucker !', status=204, mimetype='text/plain')

    chat_input = f'{speaker} : {transcription}'
    logging.info(chat_input)
    response = chat.answer(chat_input)
    logging.info(f'ChatGPT : {response}')

    intake_2.put(response)
    encoder = MultipartEncoder(fields={'text': response, 'audio': ('audio', sink_2.get().tobytes())})
    return Response(response=encoder.to_string(), status=200, mimetype=encoder.content_type)


@app.route('/video')
def video_stream():
    return 'Not yet implemented', 500


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    SAMPLE_RATE = 16000

    transcriber = VoiceTranscriber()
    synthesizer = VoiceSynthesizer()
    chat = ChatGPT()

    intake_1, sink_1 = transcriber.create_intake(), transcriber.create_sink()
    intake_2, sink_2 = synthesizer.create_intake(), synthesizer.create_sink()

    transcriber.start()
    synthesizer.start()

    app.run(host='0.0.0.0', debug=True, threaded=True, port=9999)
