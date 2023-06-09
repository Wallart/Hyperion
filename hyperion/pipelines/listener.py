from time import time
from hyperion.audio import float32_to_int16
from hyperion.gui import UIAction
from hyperion.utils.logger import ProjectLogger
from hyperion.gui.chat_window import ChatWindow
from hyperion.utils.protocol import frame_decode
from hyperion.audio.io.source.in_file import InFile
from hyperion.audio.io.audio_input import AudioInput
from hyperion.audio.io.audio_output import AudioOutput
from hyperion.audio.io.source.in_device import InDevice
from concurrent.futures import ThreadPoolExecutor
from hyperion.video.video_input import VideoInput
from hyperion.voice_processing.voice_detector import VoiceDetector
from hyperion.voice_processing.voice_recognizer import VoiceRecognizer

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

        source = InDevice(opts.in_idx, self._in_sample_rate, db=opts.db) if self._dummy_file is None else InFile(self._dummy_file, self._in_sample_rate)
        self.audio_in = AudioInput(source)
        detector = VoiceDetector(ctx, self._in_sample_rate, activation_threshold=.9)
        self.audio_out = AudioOutput(opts.out_idx, self._out_sample_rate)
        self.intake = self.audio_out.create_intake()

        if opts.recog:
            recognizer = VoiceRecognizer(ctx)
            self.sink = self.audio_in.pipe(detector).pipe(recognizer).create_sink()
            self.threads = [self.audio_in, detector, recognizer, self.audio_out]
        else:
            self.sink = self.audio_in.pipe(detector).create_sink()
            self.threads = [self.audio_in, detector, self.audio_out]

        self._requests_preprompt = None
        self._requests_llm = None
        self._camera_handler = None

        if not opts.no_gui:
            bot_name = requests.get(url=f'{self._target_url}/name').content.decode('utf-8')
            prompts = requests.get(url=f'{self._target_url}/prompts').json()
            current_prompt = requests.get(url=f'{self._target_url}/prompt').content.decode('utf-8')
            models = requests.get(url=f'{self._target_url}/models').json()
            current_model = requests.get(url=f'{self._target_url}/model').content.decode('utf-8')

            def dbs_delegate():
                return self.audio_in._source.prev_dbs

            def params_delegate():
                cam_device = -1 if self._camera_handler is None else self._camera_handler.device
                return self.audio_in._source.db_threshold, self.audio_in._source.device_name, self.audio_out.device_name, cam_device

            self._gui = ChatWindow(bot_name, prompts, current_prompt, models, current_model, title=bot_name)
            self._gui.params_delegate = params_delegate
            self._gui.dbs_delegate = dbs_delegate
            self._gui.start_threads()

            self._audio_handler = threading.Thread(target=self._audio_request_handler, daemon=False)
            self._text_handler = threading.Thread(target=self._text_request_handler, daemon=False)
            self.threads.extend([self._audio_handler, self._text_handler])

            res = requests.get(url=f'{self._target_url}/state')
            if res.status_code == 200:
                self._gui.update_status('online')
            else:
                self._gui.update_status('offline')

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

        _ = [t.join() for t in self.threads]
        if self._camera_handler is not None:
            self._camera_handler.join()
        sio.disconnect()

    def stop(self):
        self.running = False
        _ = [t.stop() for t in self.threads if hasattr(t, 'stop')]
        if self._camera_handler is not None:
            self._camera_handler.stop()

    def interrupt(self, timestamp):
        if not self._opts.no_gui:
            self._gui.mute(timestamp)
        self.audio_out.mute(timestamp)

    def _process_request(self, api_endpoint, payload, requester=None):
        t0 = time()
        headers = {'SID': self.sid}
        if self._requests_preprompt is not None:
            headers['preprompt'] = self._requests_preprompt
        if self._requests_llm is not None:
            headers['model'] = self._requests_llm

        opts = {
            'url': f'{self._target_url}/{api_endpoint}',
            'stream': True,
            'headers': headers
        }
        if api_endpoint == 'chat':
            opts['data'] = payload
        else:
            opts['files'] = payload
        try:
            res = requests.post(**opts)
            if res.status_code != 200:
                if not self._opts.no_gui:
                    if res.status_code == 418:
                        self._gui.update_status('sleeping')
                    elif res.status_code == 204:
                        self._gui.update_status('no speech detected')
                    else:
                        self._gui.update_status('error')

                ProjectLogger().warning(f'Response HTTP {res.status_code}')
                if len(res.content) > 0:
                    ProjectLogger().info(res.content)
                    return
            elif not self._opts.no_gui:
                self._gui.update_status('online')

            buffer = bytearray()
            for bytes_chunk in res.iter_content(chunk_size=4096):
                buffer.extend(bytes_chunk)
                output = frame_decode(buffer)
                if output is None:
                    continue

                decoded_frame, buffer = output
                self._distribute(decoded_frame)

            ProjectLogger().info(f'Request processed in {time() - t0:.3f} sec(s).')
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

    def _distribute(self, decoded_frame):
        idx = decoded_frame['IDX']
        timestamp = decoded_frame['TIM']
        speaker = decoded_frame['SPK']
        request = decoded_frame['REQ']
        answer = decoded_frame['ANS']
        audio = decoded_frame['PCM']

        if not self._opts.no_gui:
            self._gui.queue_message(timestamp, idx, speaker, request, answer)

        ProjectLogger().info(f'{speaker} : {request}')
        ProjectLogger().info(f'ChatGPT : {answer}')
        spoken_chunk = np.frombuffer(audio, dtype=np.int16)
        if len(spoken_chunk) > 0:
            self.intake.put((timestamp, spoken_chunk))
            # self.source.set_feedback(spoken_chunk, self._out_sample_rate)

    def _audio_request_handler(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                try:
                    data = self.sink.drain()
                    if self._recog:
                        speech_chunk, recognized_speaker = data
                        if (speech_chunk is None and recognized_speaker is None) or recognized_speaker == 'Unknown':
                            ProjectLogger().info('Request ignored.')
                            continue

                        _ = executor.submit(self._process_speech_request, recognized_speaker, speech_chunk.numpy())
                    elif data is not None:
                        _ = executor.submit(self._process_audio_request, data.numpy())
                except queue.Empty:
                    continue

        ProjectLogger().info('Audio request handler stopped.')

    def _stop_camera_handler(self):
        self._camera_handler.stop()
        self._camera_handler.join()
        self._camera_handler = None

    def _text_request_handler(self):
        while self.running:
            try:
                # one thread is enough for text
                event = self._gui.drain_message()
                if event[0] == UIAction.QUIT:
                    self.stop()
                    break
                elif event[0] == UIAction.SEND_MESSAGE:
                    username, text = event[1], event[2]
                    self._process_text_request(username, text)
                elif event[0] == UIAction.CHANGE_INPUT_DEVICE:
                    self.audio_in.change(InDevice(event[1], self._in_sample_rate, db=self.audio_in._source.db_threshold))
                elif event[0] == UIAction.CHANGE_OUTPUT_DEVICE:
                    self.audio_out.change(event[1])
                elif event[0] == UIAction.CHANGE_DB:
                    self.audio_in._source.db_threshold = event[1]
                elif event[0] == UIAction.DISABLED_CAMERA_DEVICE:
                    if self._camera_handler is not None:
                        self._stop_camera_handler()
                elif event[0] == UIAction.CHANGE_CAMERA_DEVICE:
                    if self._camera_handler is not None:
                        self._stop_camera_handler()

                    self._camera_handler = VideoInput(event[1], self._target_url)
                    self._camera_handler.start()
                    self._gui.set_camera_feedback(self._camera_handler.create_sink(maxsize=1), self._camera_handler.width, self._camera_handler.height)
                elif event[0] == UIAction.CHANGE_PROMPT:
                    self._requests_preprompt = event[1]
                elif event[0] == UIAction.CHANGE_LLM:
                    self._requests_llm = event[1]
            except queue.Empty:
                continue

        ProjectLogger().info('Text request handler stopped.')
