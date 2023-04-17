from PIL import Image
from time import time
from hyperion.utils.threading import Consumer
from hyperion.utils.logger import ProjectLogger
from lavis.models import load_model_and_preprocess

import queue


class VisualQuestionAnswering(Consumer):

    def __init__(self, ctx, gpt_delegate):
        super().__init__()
        self._ctx = ctx
        opts = {
            'name': 'blip_caption',
            'model_type': 'base_coco',
            'is_eval': True,
            'device': ctx[-1]
        }
        self.model, self.vis_processors, _ = load_model_and_preprocess(**opts)
        self._gpt_delegate = gpt_delegate

    def run(self) -> None:
        while self.running:
            try:
                frame = self._consume()
                t0 = time()

                image = Image.fromarray(frame)
                processed_image = self.vis_processors['eval'](image).unsqueeze(0).to(self._ctx[-1])
                caption = self.model.generate({'image': processed_image})[0]
                self._gpt_delegate.add_video_context(caption)
                ProjectLogger().info(caption)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
