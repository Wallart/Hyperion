from brain.chat_gpt import ChatGPT
from flask import Flask, Response, request
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import logging
import numpy as np

app = Flask(__name__)

TEXT_SEPARATOR = b'----TEXT-END----\n'
CHUNK_SEPARATOR = b'----CHUNK-END----\n'


def sink_streamer(request, text_sink, audio_sink):
    i = 0
    while True:
        text_chunk = text_sink.get()
        audio_chunk = audio_sink.get()
        if audio_chunk is None:
            return

        req = bytes(request, 'utf-8')
        resp = bytes(text_chunk, 'utf-8')
        audio_bytes = audio_chunk.tobytes()
        response = resp + TEXT_SEPARATOR + audio_bytes + CHUNK_SEPARATOR
        response = req + TEXT_SEPARATOR + response if i == 0 else response
        i += 1
        yield response


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

    intake_2.put(chat_input)
    return Response(response=sink_streamer(transcription, sink_2a, sink_2b), status=200, mimetype='application/octet-stream')


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
    intake_2, sink_2a, sink_2b = chat.create_intake(), chat.create_sink(), chat.pipe(synthesizer).create_sink()

    transcriber.start()
    synthesizer.start()
    chat.start()

    app.run(host='0.0.0.0', debug=False, threaded=True, port=9999)
