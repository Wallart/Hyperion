from time import time
from PIL import Image
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.threading import Producer, Consumer
from hyperion.utils.request import RequestObject, KeepAliveSet
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import io
import queue
import torch


class ImageGenerator(Consumer, Producer):
    def __init__(self, ctx, model_name='stabilityai/stable-diffusion-2-1-base'):
        super().__init__()
        self._ctx = ctx

        self.synthesizer_intake = None

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self._pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe = self._pipe.to(ctx[0])

    def set_synthesizer_intake(self, synthesizer_intake_delegate):
        self.synthesizer_intake = synthesizer_intake_delegate

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def flush_img(self, image, request_obj, is_terminal, text=None):
        bytes_arr = io.BytesIO()
        image.save(bytes_arr, format='jpeg')
        new_request_obj = RequestObject.copy(request_obj)
        new_request_obj.image_answer = bytes_arr.getvalue()
        new_request_obj.priority = request_obj.num_answer
        if new_request_obj.text_answer is None and text is not None:
            new_request_obj.text_answer = text

        self._forward(new_request_obj, is_terminal)

    def _forward(self, request, terminal_node):
        if terminal_node:
            self._put(request, request.identifier)
        else:
            self.synthesizer_intake.put(request)

    def run(self) -> None:
        while self.running:
            try:
                request_obj = self._consume()
                t0 = time()

                is_terminal = True if request_obj.text_answer is None else False

                ack = RequestObject.copy(request_obj)
                ack.text_answer = '<ACK>'
                ack.silent = True
                self._forward(ack, is_terminal)

                request_obj.priority = request_obj.num_answer
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
                        self.flush_img(grid, request_obj, is_terminal)
                    else:
                        for i, image in enumerate(images):
                            text = f'Image #{i + 1}' if len(images) > 1 else ''
                            self.flush_img(image, request_obj, is_terminal, text)
                            request_obj.num_answer += 1

                except RuntimeError as e:
                    ProjectLogger().error(e)

                    err = RequestObject.copy(request_obj)
                    err.text_answer = '<ERR>'
                    err.silent = True
                    err.priority = 0
                    self._forward(err, is_terminal)

                    err_response = RequestObject.copy(request_obj)
                    err_response.text_answer = str(e)
                    err_response.silent = True
                    self._forward(err_response, is_terminal)

                # termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)
                term_req = KeepAliveSet().remove(request_obj.identifier)
                if term_req is not None:
                    self._forward(term_req, is_terminal)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} exec. time')
            except queue.Empty:
                continue
