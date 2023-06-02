from time import time
from PIL import Image
from copy import deepcopy
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.request import RequestObject
from hyperion.utils.threading import Producer, Consumer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import io
import queue
import torch


class ImageGenerator(Consumer, Producer):
    def __init__(self, ctx, model_name='stabilityai/stable-diffusion-2-1-base'):
        super().__init__()
        self._ctx = ctx

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self._pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe = self._pipe.to(ctx[-1])

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def flush_img(self, image, request_obj, text=''):
        bytes_arr = io.BytesIO()
        image.save(bytes_arr, format='jpeg')
        new_request_obj = deepcopy(request_obj)
        new_request_obj.image_answer = bytes_arr.getvalue()
        new_request_obj.text_answer = text
        new_request_obj.priority = request_obj.num_answer
        self._put(new_request_obj, new_request_obj.identifier)

    def run(self) -> None:
        while self.running:
            try:
                request_obj = self._consume()
                t0 = time()

                ack = deepcopy(request_obj)
                ack.text_answer = 'Processing...\n'
                request_obj.priority = request_obj.num_answer
                self._put(ack, request_obj.identifier)
                request_obj.num_answer += 1

                cmd_args = request_obj.command_args
                batch = 1 if cmd_args['batch'] is None else cmd_args['batch']
                images_prompts = [cmd_args['sentence']] * batch
                mosaic = cmd_args['mosaic']

                rows, cols = batch, 1
                exkeys = ['sentence', 'mosaic', 'batch']
                args = {k: v for k, v in cmd_args.items() if k not in exkeys and v is not None}

                try:
                    images = self._pipe(images_prompts, **args).images
                    if mosaic and batch > 1:
                        grid = ImageGenerator.image_grid(images, rows, cols)
                        self.flush_img(grid, request_obj)
                    else:
                        # TODO Shared sink is a bad idea
                        for i, image in enumerate(images):
                            self.flush_img(image, request_obj, f'Image #{i + 1}')
                            request_obj.num_answer += 1

                except RuntimeError as e:
                    ProjectLogger().error(e)
                    err_response = deepcopy(request_obj)
                    err_response.text_answer = str(e)
                    self._put(err_response, request_obj.identifier)

                termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True, priority=999)
                self._put(termination_request, request_obj.identifier)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
