from brain.chat_gpt import ChatGPT
from flask import Flask, Response, request
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_synthesizer import VoiceSynthesizer
from voice_processing.voice_transcriber import VoiceTranscriber

import logging
import numpy as np

app = Flask(__name__)


@app.route('/audio', methods=['POST'])
def audio_stream():
    byte_array = request.data
    audio_chunk = np.frombuffer(byte_array, dtype=np.float32)
    intake_1.put(audio_chunk)
    transcription = sink_1.get()
    response = ...
    if transcription is None:
        # transcription = 'Ce que raconte l\'utilisateur est à peine audible, et Hypérion ne comprend pas.'
        # response = chat.answer(transcription, 'system')
        return Response(response='Speak louder motherfucker !', status=204, mimetype='text/plain')
    else:
        logging.info(f'User : {transcription}')
        response = chat.answer(transcription)

    intake_2.put(response)
    wav = sink_2.get()
    return Response(response=wav.tobytes(), status=200, mimetype='application/octet-stream')


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
