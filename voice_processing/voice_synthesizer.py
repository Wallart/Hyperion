from utils.threading import Consumer, Producer
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
# from TTS.api import TTS

# import numpy as np


class VoiceSynthesizer(Consumer, Producer):

    def __init__(self):
        super().__init__()

        # model_list = TTS.list_models()
        # print(model_list)
        # model_name = model_list[0]
        # self._tts = TTS(model_name)
        # print(self._tts.languages)
        # self.sample_rate = 16000

        models, cfg, self._task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
        self._model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self._task.data_cfg)
        self._generator = self._task.build_generator(models, cfg)
        self.sample_rate = 22050

    def run(self) -> None:
        while True:
            text = self._in_queue.get()

            # Run TTS
            # ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
            # Text to speech with a numpy output
            # wav = self._tts.tts(text, speaker=self._tts.speakers[0], language='fr-fr')
            # self._dispatch(np.array(wav, dtype=np.float32))
            sample = TTSHubInterface.get_model_input(self._task, text)
            wav, sample_rate = TTSHubInterface.get_prediction(self._task, self._model, self._generator, sample)
            self._dispatch(wav.numpy())


if __name__ == '__main__':
    import sounddevice as sd
    import numpy as np
    # models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    # model = models[0]
    # TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    # generator = task.build_generator(models, cfg)

    # text = 'ceci est un test.'
    # sample = TTSHubInterface.get_model_input(task, text)
    # wav, sample_rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    model_list = TTS.list_models()
    print(model_list)
    model_name = model_list[0]
    tts = TTS(model_name)
    sample_rate = 16000

    speaker = sd.OutputStream(sample_rate, channels=1, dtype=np.float32)
    speaker.start()
    for sp in tts.speakers:
        wav = tts.tts('Bonjour ceci est un putain de test de fou furieux !', speaker=sp, language='fr-fr', speaker_wav='/Users/wallart/Desktop/test.wav')
        speaker.write(np.array(wav, dtype=np.float32))
    # speaker.write(wav.numpy())
    speaker.stop()
