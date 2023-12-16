from time import time
from gtts import gTTS
from TTS.api import TTS
from hyperion.utils import load_file
from hyperion.audio import float32_to_int16
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.protocol import frame_encode
from hyperion.voice_processing import download_model
from hyperion.utils.identity_store import IdentityStore
from hyperion.utils.threading import Consumer, Producer
from elevenlabs import set_api_key, voices, generate, RateLimitError

import os
import io
import queue
import pydub
import numpy as np
import google.cloud.texttospeech as tts

VALID_ENGINES = ['local', 'eleven', 'google_cloud', 'google_translate']


class VoiceSynthesizer(Consumer, Producer):

    def __init__(self, ctx, sio_delegate):
        super().__init__()
        self.sio = sio_delegate
        self.sample_rate = 24000

        eleven_key_path = ProjectPaths().resources_dir / 'keys' / 'elevenlabs_api.key'
        google_key_path = ProjectPaths().resources_dir / 'keys' / 'google_api.key'

        self._init_local_model(ctx)
        self._preferred_engines = [VALID_ENGINES[0]]

        if eleven_key_path.exists() or 'ELEVENLABS_API' in os.environ:
            eleven_api_key = load_file(eleven_key_path)[0] if eleven_key_path.exists() else os.environ['ELEVENLABS_API']
            self._init_elevenlabs(eleven_api_key)
            self._preferred_engines.append(VALID_ENGINES[1])

        if google_key_path.exists() or 'GOOGLE_API' in os.environ:
            google_api_key = load_file(google_key_path)[0] if google_key_path.exists() else os.environ['GOOGLE_API']
            try:
                self._init_google_cloud_synth(google_api_key)
                self._preferred_engines.append(VALID_ENGINES[2])
            except Exception as e:
                ProjectLogger().warning(e.message)

        self._preferred_engines.append(VALID_ENGINES[3])

    def get_preferred_engines(self):
        return self._preferred_engines

    def set_preferred_engines(self, engines):
        for e in engines:
            if e not in VALID_ENGINES:
                return False

        self._preferred_engines = engines
        return True

    def get_engine_valid_voices(self, engine):
        if engine == 'google_cloud':
            return self._valid_google_voices
        elif engine == 'eleven':
            return self._valid_eleven_voices
        elif engine == 'local':
            return self._valid_local_voices
        return False

    def get_engine_default_voice(self, engine):
        if engine == 'google_cloud':
            return self._default_google_voice
        elif engine == 'eleven':
            return self._default_eleven_voice
        elif engine == 'local':
            return self._default_local_voice
        return False

    def set_engine_default_voice(self, engine, voice):
        if engine == 'google_cloud' and voice in self._valid_google_voices:
            self._default_google_voice = voice
            return True
        elif engine == 'eleven' and voice in self._valid_eleven_voices:
            self._default_eleven_voice = voice
            return True
        elif engine == 'local' and voice in self._valid_local_voices:
            self._default_local_voice = voice
            return True
        return False

    def _init_elevenlabs(self, api_key):
        set_api_key(api_key)
        self._default_eleven_voice = 'Josh'
        self._valid_eleven_voices = [v.name for v in voices()]

    def _init_google_cloud_synth(self, api_key):
        self._language_code = 'fr-FR'
        self._default_google_voice = 'fr-FR-Neural2-B'
        self._client = tts.TextToSpeechClient(client_options={'api_key': api_key})
        voices = self._client.list_voices().voices
        self._valid_google_voices = [v.name for v in voices if v.language_codes[0] == self._language_code]

    def _init_local_model(self, ctx):
        self._sample_dir = ProjectPaths().resources_dir / 'speakers_samples'
        self._default_local_voice = 'tim'
        self._valid_local_voices = [e.name for e in self._sample_dir.glob('*') if e.is_dir()]

        model_name = 'xtts_v2.0.2'
        download_model(model_name)
        self._local_tts = TTS(model_name).to(ctx[0])

    def _eleven_synthesizer(self, text, voice=None):
        voice_name = self._default_eleven_voice if voice is None or voice not in self._valid_eleven_voices else voice
        audio = generate(
            text=text,
            stream=False,
            voice=voice_name,
            model='eleven_multilingual_v1'
        )

        mp3_sound = pydub.AudioSegment.from_file(io.BytesIO(audio), format='mp3')
        memory_buff = io.BytesIO()
        mp3_sound.export(memory_buff, format='wav')
        wav_sound = pydub.AudioSegment.from_wav(memory_buff)
        wav_sound = wav_sound.set_frame_rate(self.sample_rate)
        wav_array = np.array(wav_sound.get_array_of_samples(), dtype=np.int16)
        return wav_array

    def _google_cloud_synthesizer(self, text, voice=None):
        voice_name = self._default_google_voice if voice is None or voice not in self._valid_google_voices else voice
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(language_code=self._language_code, name=voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, speaking_rate=1.2)
        response = self._client.synthesize_speech(input=text_input, voice=voice_params, audio_config=audio_config)

        wav_array = np.frombuffer(response.audio_content, dtype=np.int16)
        return wav_array[1000:]  # remove popping sound

    def _google_translate_synthesizer(self, text, **kwargs):
        tts = gTTS(text, lang='fr', slow=False)
        data = None
        for raw_buffer in tts.stream():
            buffer = np.frombuffer(raw_buffer, dtype=np.float32)
            data = buffer if data is None else np.concatenate([data, buffer])

        mp3 = pydub.AudioSegment.from_file(io.BytesIO(data), format='mp3')

        memory_buff = io.BytesIO()
        mp3.export(memory_buff, format='wav')
        sound = pydub.AudioSegment.from_wav(memory_buff)
        wav_array = np.array(sound.get_array_of_samples(), dtype=np.int16)
        return wav_array

    def _local_synthesizer(self, text, voice=None):
        voice_name = self._default_local_voice if voice is None or voice not in self._valid_local_voices else voice
        samples = list((self._sample_dir / voice_name).glob('*.wav'))
        wav_sound = self._local_tts.tts(text=text, speaker_wav=samples, language='fr')
        wav = np.array(wav_sound, dtype=np.float32)
        return float32_to_int16(wav)

    def _infer(self, text, engine=None, voice=None):
        if engine is None:
            for e in self._preferred_engines:
                try:
                    return self._infer_with_engine(text, e, voice)
                except RateLimitError as e:
                    ProjectLogger().error(e)
        else:
            return self._infer_with_engine(text, engine, voice)

    def _infer_with_engine(self, text, engine, voice):
        if engine == 'eleven':
            return self._eleven_synthesizer(text, voice)
        elif engine == 'google_cloud':
            return self._google_cloud_synthesizer(text, voice)
        elif engine == 'local':
            return self._local_synthesizer(text)
        else:
            return self._google_translate_synthesizer(text)

    def run(self) -> None:
        while self.running:
            try:
                request_obj = self._consume()
                if request_obj.termination:
                    self._put(request_obj, request_obj.identifier)
                    continue

                t0 = time()
                if not request_obj.silent:
                    ProjectLogger().info(f'Synthesizing speech...')
                    try:
                        wav = self._infer(request_obj.text_answer, engine=request_obj.speech_engine, voice=request_obj.voice)
                        request_obj.audio_answer = wav
                    except Exception as e:
                        ProjectLogger().error(f'Synthesizer muted : {e}')
                else:
                    ProjectLogger().info(f'Silent answer requested.')

                if request_obj.push:
                    if request_obj.identifier in IdentityStore().inverse and request_obj.identifier is not None:
                        socket_ids = IdentityStore().inverse[request_obj.identifier]
                        args = [
                            request_obj.timestamp,
                            request_obj.num_answer,
                            request_obj.user,
                            request_obj.text_request,
                            request_obj.text_answer,
                            request_obj.audio_answer,
                            request_obj.image_answer
                        ]
                        _ = [self.sio().emit('data', frame_encode(*args), to=socket_id) for socket_id in socket_ids]
                else:
                    self._put(request_obj, request_obj.identifier)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Synthesizer stopped.')
