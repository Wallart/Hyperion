from enum import Enum
from time import time
from hyperion.utils.timer import Timer
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.request import RequestObject
from hyperion.utils.threading import Consumer, Producer

import json
import shlex
import queue
import argparse


class ACTIONS(Enum):
    SLEEP = 0
    WAKE = 1
    WIPE = 2
    QUIET = 3
    DRAW = 4


class UserCommandDetector(Consumer, Producer):
    def __init__(self, clear_ctx_delegate, sio_delegate):
        super().__init__()
        self.frozen = False
        self.sio = sio_delegate
        self.img_intake = None
        self.clear_context = clear_ctx_delegate

        self._commands_file = ProjectPaths().resources_dir / 'default_sentences' / 'user_commands.json'

        with open(self._commands_file) as f:
            self._commands = json.load(f)

    def set_img_intake(self, img_intake_delegate):
        self.img_intake = img_intake_delegate

    def run(self):
        while self.running:
            try:
                request_obj = self._consume()
                t0 = time()

                analyzed_text = request_obj.text_request.lower()
                action = None
                for i, sentences in enumerate(self._commands.values()):
                    for sentence in sentences:
                        tokens = sentence.lower().split(' ')
                        found_tokens = sum([1 for t in tokens if t in analyzed_text])
                        if found_tokens == len(tokens):
                            action = i
                            break
                    if action is not None:
                        break

                if action is None and not self.frozen:
                    self._dispatch(request_obj)
                elif action is not None:
                    ProjectLogger().info(f'Command found in "{analyzed_text}"')
                    termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)

                    if action == ACTIONS.WAKE.value:
                        self._on_wake_up(request_obj, termination_request)
                    elif not self.frozen:
                        if action == ACTIONS.SLEEP.value:
                            self._on_sleep(request_obj, termination_request)
                        elif action == ACTIONS.WIPE.value:
                            self._on_memory_wipe(request_obj, termination_request)
                        elif action == ACTIONS.QUIET.value:
                            self._on_quiet(request_obj, termination_request)
                        elif action == ACTIONS.DRAW.value:
                            self._on_draw(request_obj)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} COMMAND exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Command Detector stopped.')

    def _on_draw(self, request_obj):
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch', type=int)
        parser.add_argument('-W', '--width', type=int)
        parser.add_argument('-H', '--height', type=int)
        parser.add_argument('-s', '--steps', dest='num_inference_steps', type=int)
        parser.add_argument('-g', '--guidance-scale', type=float)
        parser.add_argument('-m', '--mosaic', action='store_true')
        parser.add_argument('sentence', type=str)

        try:
            command_line = request_obj.text_request.split(self._commands['DRAW'][0])[-1].strip()
            args = parser.parse_args(shlex.split(command_line))
            for k, v in vars(args).items():
                request_obj.command_args[k] = v

            self.img_intake.put(request_obj)
        except (SystemExit, ValueError) as e:
            err = RequestObject.copy(request_obj)
            err.text_answer = '<ERR>'
            err.silent = True
            err.priority = 0
            self._put(err, request_obj.identifier)

            err_request = RequestObject.copy(request_obj)
            err_request.text_answer = 'Invalid arguments'
            err_request.silent = True
            self._put(err_request, request_obj.identifier)

            termination = RequestObject(request_obj.identifier, request_obj.user, termination=True)
            termination.priority = 2
            self._put(termination, request_obj.identifier)

    def _on_quiet(self, request_obj, termination_request):
        termination_request.priority = 0
        self._put(termination_request, request_obj.identifier)
        self.sio().emit('interrupt', Timer().now(), to=request_obj.socket_id)

    def _on_sleep(self, request_obj, termination_request):
        self.frozen = True

        ack = RequestObject.copy(request_obj)
        ack.text_answer = '<SLEEPING>'
        ack.silent = True

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)

    def _on_wake_up(self, request_obj, termination_request):
        self.frozen = False

        ack = RequestObject.copy(request_obj)
        ack.text_answer = '<WAKE>'
        ack.silent = True

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)

    def _on_memory_wipe(self, request_obj, termination_request):
        self.clear_context(request_obj.preprompt)

        ack = RequestObject.copy(request_obj)
        ack.text_answer = '<MEMWIPE>'
        ack.silent = True

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)
