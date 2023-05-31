from time import time
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

        self._default_width = 640
        self._default_height = 480

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self._pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe = self._pipe.to(ctx[-1])

    def run(self) -> None:
        while self.running:
            try:
                request_obj = self._consume()
                t0 = time()

                ack = deepcopy(request_obj)
                ack.text_answer = 'Processing...'
                # self._dispatch(ack)
                self._put(ack, request_obj.identifier)

                args = request_obj.command_args
                batch = 1 if args['batch'] is None else args['batch']
                images_prompts = [args['sentence']] * batch
                width = self._default_width if args['width'] is None else args['width']
                height = self._default_height if args['height'] is None else args['height']

                try:
                    images = self._pipe(images_prompts, width=width, height=height).images
                    for i, image in enumerate(images):
                        bytes_arr = io.BytesIO()
                        image.save(bytes_arr, format='jpeg')
                        request_obj.image_answer = bytes_arr.getvalue()
                        request_obj.text_answer = ''
                        request_obj.num_answer = 1
                        # self._dispatch(request_obj)
                        self._put(request_obj, request_obj.identifier)
                        if not args['mosaic']:
                            break
                except RuntimeError as e:
                    ProjectLogger().error(e)
                    err_response = deepcopy(request_obj)
                    err_response.text_answer = str(e)
                    self._put(err_response, request_obj.identifier)

                termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)
                self._put(termination_request, request_obj.identifier)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
