from time import time
from gtts import gTTS
from hyperion.utils import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from elevenlabs import set_api_key, voices, generate
from hyperion.utils.threading import Consumer, Producer

import io
import queue
import torch
import pydub
import numpy as np
import google.cloud.texttospeech as tts

VALID_ENGINES = ['eleven', 'google_cloud', 'google_translate']


class VoiceSynthesizer(Consumer, Producer):

    def __init__(self):
        super().__init__()

        self.sample_rate = 24000
        self._preferred_engines = []

        eleven_key_path = ProjectPaths().resources_dir / 'keys' / 'elevenlabs_api.key'
        google_key_path = ProjectPaths().resources_dir / 'keys' / 'google_api.key'

        if eleven_key_path.exists():
            with open(eleven_key_path) as f:
                eleven_api_key = f.readlines()[0]
            self._init_elevenlabs(eleven_api_key)
            self._preferred_engines.append('eleven')

        if google_key_path.exists():
            with open(google_key_path) as f:
                google_api_key = f.readlines()[0]
            self._init_google_cloud_synth(google_api_key)
            self._preferred_engines.append('google_cloud')

        self._preferred_engines.append('google_translate')

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
        return False

    def get_engine_default_voice(self, engine):
        if engine == 'google_cloud':
            return self._default_google_voice
        elif engine == 'eleven':
            return self._default_eleven_voice
        return False

    def set_engine_default_voice(self, engine, voice):
        if engine == 'google_cloud' and voice in self._valid_google_voices:
            self._default_google_voice = voice
            return True
        elif engine == 'eleven' and voice in self._valid_eleven_voices:
            self._default_eleven_voice = voice
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

    def _init_local_model(self):
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
        from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub

        # TODO Use librosa to resample to 24000 Hz
        self.sample_rate = 22050
        self._gpu = torch.device('cuda:0')

        models, cfg, self._task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
        self._model = models[0].to(self._gpu)
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self._task.data_cfg)
        self._generator = self._task.build_generator(models, cfg)

    def _local_model(self, text):
        # FairSeq
        sample = TTSHubInterface.get_model_input(self._task, text)
        sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(self._gpu)
        sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(self._gpu)
        sample['speaker'] = sample['speaker'].to(self._gpu)

        wav, sample_rate = TTSHubInterface.get_prediction(self._task, self._model, self._generator, sample)
        wav = wav.cpu().numpy()
        return wav

    def _infer(self, text, engine=None, voice=None):
        if engine is None:
            for e in self._preferred_engines:
                try:
                    return self._infer_with_engine(text, e, voice)
                except e:
                    pass
        else:
            return self._infer_with_engine(text, engine, voice)

    def _infer_with_engine(self, text, engine, voice):
        if engine == 'eleven':
            return self._eleven_synthesizer(text, voice)
        elif engine == 'google_cloud':
            return self._google_cloud_synthesizer(text, voice)
        else:
            return self._google_translate_synthesizer(text)

    def run(self) -> None:
        while self.running:
            try:
                request_obj = self._consume()
                if request_obj.termination:
                    self._put(request_obj, request_obj.identifier)
                    continue

                ProjectLogger().info(f'Synthesizing speech...')
                t0 = time()

                try:
                    wav = self._infer(request_obj.text_answer, engine=request_obj.speech_engine, voice=request_obj.voice)
                    request_obj.audio_answer = wav
                except Exception as e:
                    ProjectLogger().error(f'Synthesizer muted : {e}')

                self._put(request_obj, request_obj.identifier)
                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Synthesizer stopped.')
