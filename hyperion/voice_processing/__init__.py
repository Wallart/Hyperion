from TTS.api import TTS
from pathlib import Path
from TTS.utils.manage import ModelManager

import os


def download_model(model_name):
    manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=True, verbose=False)
    model_item, model_full_name, model, md5sum = manager._set_model_item(model_name)
    model_fullpath = Path(manager.output_prefix) / model_full_name

    tos_path = model_fullpath / 'tos_agreed.txt'
    if not tos_path.exists():
        os.makedirs(model_fullpath, exist_ok=True)
        with open(tos_path, 'w', encoding='utf-8') as f:
            f.write('I have read, understood and agreed to the Terms and Conditions.')

        manager._download_hf_model(model_item, model_fullpath)
