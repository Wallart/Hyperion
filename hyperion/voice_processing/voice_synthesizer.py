from time import time
from gtts import gTTS
from hyperion.utils import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.threading import Consumer, Producer

import io
import queue
import torch
import numpy as np
import google.cloud.texttospeech as tts


class VoiceSynthesizer(Consumer, Producer):

    def __init__(self):
        super().__init__()

        mode = 'cloud'
        key_path = ProjectPaths().resources_dir / 'keys' / 'google_api.key'
        if key_path.exists():
            with open(key_path) as f:
                api_key = f.readlines()[0]
        else:
            mode = 'translate'

        if mode == 'cloud':
            self._init_google_cloud_synth(api_key)
            self._infer = self._google_cloud_synthesizer
        elif mode == 'local':
            self._init_local_model()
            self._infer = self._local_model
        elif mode == 'translate':
            self._init_google_translate_synth()
            self._infer = self._google_translate_synthesizer
        else:
            raise Exception('Unknown speech synthesizer option')

    def _init_google_cloud_synth(self, api_key):
        self.sample_rate = 24000
        self._language_code = 'fr-FR'
        self._voice_name = 'fr-FR-Neural2-B'

        self._client = tts.TextToSpeechClient(client_options={'api_key': api_key})

    def _google_cloud_synthesizer(self, text):
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(language_code=self._language_code, name=self._voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, speaking_rate=1.2)
        response = self._client.synthesize_speech(input=text_input, voice=voice_params, audio_config=audio_config)

        wav_array = np.frombuffer(response.audio_content, dtype=np.int16)
        return wav_array[1000:]  # remove popping sound

    def _init_google_translate_synth(self):
        self.sample_rate = 24000

    def _google_translate_synthesizer(self, text):
        import pydub
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
                    wav = self._infer(request_obj.text_answer)
                    request_obj.audio_answer = wav
                except Exception as e:
                    ProjectLogger().error(f'Synthesizer muted : {e}')

                self._put(request_obj, request_obj.identifier)
                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Synthesizer stopped.')
