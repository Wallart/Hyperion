from PIL import Image
from time import time
from typing import List
from hyperion.utils.logger import ProjectLogger
from lavis.models import load_model_and_preprocess
from hyperion.utils.threading import Consumer, Producer

import queue


class VisualQuestionAnswering(Consumer, Producer):

    def __init__(self, ctx):
        super().__init__()
        self._ctx = ctx
        opts = {
            'name': 'blip_caption',
            'model_type': 'base_coco',
            'is_eval': True,
            'device': ctx[0]
        }
        self.model, self.vis_processors, _ = load_model_and_preprocess(**opts)
        self.chat_delegate = None

    def set_chat_delegate(self, chat_delegate):
        self.chat_delegate = chat_delegate

    def analyze_frame(self, frame: List):
        image = Image.fromarray(frame)
        processed_image = self.vis_processors['eval'](image).unsqueeze(0).to(self._ctx[-1])
        # use_nucleus_sampling workaround to use Blip with transformers>4.25.0
        # https://github.com/salesforce/LAVIS/issues/142
        caption = self.model.generate({'image': processed_image}, use_nucleus_sampling=True)[0]
        return caption

    def run(self) -> None:
        while self.running:
            try:
                frame = self._consume()
                t0 = time()

                caption = self.analyze_frame(frame)
                self.chat_delegate.add_video_context(caption)
                self._dispatch(caption)

                ProjectLogger().info(caption)
                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
