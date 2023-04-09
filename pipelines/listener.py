from time import time
from audio import float32_to_int16
from utils.logger import ProjectLogger
from gui.chat_window import ChatWindow
from utils.protocol import frame_decode
from audio.io.source.in_file import InFile
from audio.io.audio_input import AudioInput
from audio.io.audio_output import AudioOutput
from audio.io.source.in_device import InDevice
from concurrent.futures import ThreadPoolExecutor
from voice_processing.voice_detector import VoiceDetector
from voice_processing.voice_recognizer import VoiceRecognizer

import os
import queue
import threading
import requests
import numpy as np


class Listener:

    def __init__(self, ctx, opts):
        self._ctx = ctx
        self._opts = opts
        self.sid = None
        self.running = False

        self._in_sample_rate = 16000
        self._out_sample_rate = 24000
        self._target_url = f'http://{opts.target_url}'
        self._dummy_file = opts.dummy_file if opts.dummy_file is None else os.path.expanduser(opts.dummy_file)
        self._recog = opts.recog

        res = requests.get(url=f'{self._target_url}/name')
        self._bot_name = res.content.decode('utf-8')

        source = InDevice(opts.in_idx, self._in_sample_rate, rms=opts.rms) if self._dummy_file is None else InFile(self._dummy_file, self._in_sample_rate)
        audio_in = AudioInput(source)
        detector = VoiceDetector(ctx, self._in_sample_rate, activation_threshold=.9)
        self.audio_out = AudioOutput(opts.out_idx, self._out_sample_rate)
        self.intake = self.audio_out.create_intake()

        if opts.recog:
            recognizer = VoiceRecognizer(ctx)
            self.sink = audio_in.pipe(detector).pipe(recognizer).create_sink()
            self.threads = [audio_in, detector, recognizer, self.audio_out]
        else:
            self.sink = audio_in.pipe(detector).create_sink()
            self.threads = [audio_in, detector, self.audio_out]

        if not opts.no_gui:
            self._gui = ChatWindow(self._bot_name, self._bot_name)
            self._audio_handler = threading.Thread(target=self._audio_request_handler, daemon=False)
            self._text_handler = threading.Thread(target=self._text_request_handler, daemon=False)
            self.threads.extend([self._audio_handler, self._text_handler])

    def start(self, sio):
        self.running = True
        try:
            _ = [t.start() for t in self.threads]
            if self._opts.no_gui:
                self._audio_request_handler()
            else:
                self._gui.mainloop()
        except KeyboardInterrupt as interrupt:
            self.stop()

        # _ = [t.join() for t in self.threads]
        _ = [t.join() for t in self.threads[1:]]
        sio.disconnect()

    def stop(self):
        self.running = False
        _ = [t.stop() for t in self.threads if hasattr(t, 'stop')]

    def interrupt(self):
        self._gui.mute()
        self.audio_out.mute()

    def _process_request(self, api_endpoint, payload, requester=None):
        t0 = time()
        opts = {
            'url': f'{self._target_url}/{api_endpoint}',
            'stream': True,
            'headers': {'SID': self.sid}
        }
        if api_endpoint == 'chat':
            opts['data'] = payload
        else:
            opts['files'] = payload
        try:
            res = requests.post(**opts)
            if res.status_code != 200:
                ProjectLogger().warning(f'Something went wrong. HTTP {res.status_code}')
                ProjectLogger().error(res.content)
                return

            # TODO Ugly should be added in communication protocol
            requester = res.headers['Speaker'] if requester is None else requester

            buffer = bytearray()
            for bytes_chunk in res.iter_content(chunk_size=4096):
                buffer.extend(bytes_chunk)
                output = frame_decode(buffer)
                if output is None:
                    continue

                decoded_frame, buffer = output
                self._distribute(requester, decoded_frame)

            ProjectLogger().info(f'{requester}\'s request processed in {time() - t0:.3f} sec(s).')
        except Exception as e:
            ProjectLogger().warning(f'Request canceled : {e}')

    def _process_audio_request(self, audio):
        audio = float32_to_int16(audio)
        payload = [
            ('audio', ('audio', audio.tobytes(), 'application/octet-stream'))
        ]
        self._process_request('audio', payload)

    def _process_speech_request(self, recognized_speaker, speech):
        speech = float32_to_int16(speech)
        ProjectLogger().info(f'Processing {recognized_speaker}\'s request...')
        payload = [
            ('speaker', ('speaker', recognized_speaker, 'text/plain')),
            ('speech', ('speech', speech.tobytes(), 'application/octet-stream'))
        ]
        self._process_request('speech', payload, requester=recognized_speaker)

    def _process_text_request(self, author, text):
        payload = {
            'user': author,
            'message': text
        }
        self._process_request('chat', payload, requester=author)

    def _distribute(self, recognized_speaker, decoded_frame):
        idx = decoded_frame['IDX']
        request = decoded_frame['REQ']
        answer = decoded_frame['ANS']
        audio = decoded_frame['PCM']

        if not self._opts.no_gui:
            self._gui.queue_message(idx, recognized_speaker, request, answer)

        ProjectLogger().info(f'{recognized_speaker} : {request}')
        ProjectLogger().info(f'ChatGPT : {answer}')
        spoken_chunk = np.frombuffer(audio, dtype=np.int16)
        if len(spoken_chunk) > 0:
            self.intake.put((idx, spoken_chunk))
            # self.source.set_feedback(spoken_chunk, self._out_sample_rate)

    def _audio_request_handler(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                try:
                    data = self.sink.drain()
                    if self._recog:
                        speech_chunk, recognized_speaker = data
                        if recognized_speaker == 'Unknown':
                            ProjectLogger().info('Request ignored.')
                            continue

                        _ = executor.submit(self._process_speech_request, recognized_speaker, speech_chunk.numpy())
                    else:
                        _ = executor.submit(self._process_audio_request, data.numpy())
                except queue.Empty:
                    continue

        ProjectLogger().info('Audio request handler stopped.')

    def _text_request_handler(self):
        while self.running:
            try:
                # one thread is enough for text
                event = self._gui.drain_message()
                if event is None:
                    self.stop()
                    break

                username, text = event
                self._process_text_request(username, text)
            except queue.Empty:
                continue

        ProjectLogger().info('Text request handler stopped.')