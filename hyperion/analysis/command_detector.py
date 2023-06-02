from enum import Enum
from time import time
from copy import deepcopy
from hyperion.utils.timer import Timer
from hyperion.utils import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from concurrent.futures import ThreadPoolExecutor
from hyperion.utils.request import RequestObject
from hyperion.utils.threading import Consumer, Producer
from openai.embeddings_utils import cosine_similarity, get_embedding, get_embeddings

import json
import shlex
import queue
import argparse
import numpy as np


class ACTIONS(Enum):
    SLEEP = 0
    WAKE = 1
    WIPE = 2
    QUIET = 3
    DRAW = 4


class CommandDetector(Consumer, Producer):
    def __init__(self, clear_ctx_delegate, sio_delegate, img_intake_delegate, threshold=.8):
        super().__init__()
        self.frozen = False
        self.sio = sio_delegate
        self.img_intake = img_intake_delegate
        self.clear_context = clear_ctx_delegate

        self._commands_file = ProjectPaths().resources_dir / 'default_sentences' / 'commands.json'

        with open(self._commands_file) as f:
            self._commands = json.load(f)

        # self._classif_threshold = threshold
        # self._model = 'text-embedding-ada-002'
        # self._labels = [
        #     'Stopper',
        #     'Veille',
        #     'Réveil',
        #     'Effacer',
        #     'Autre'
        # ]
        # self._labels_embeddings = get_embeddings(self._labels, engine=self._model)

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
                else:
                    ProjectLogger().info(f'Command found in "{analyzed_text}"')
                    termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)

                    if action == ACTIONS.WAKE.value:
                        self.frozen = False
                        self._put(termination_request, request_obj.identifier)
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
        except SystemExit as e:
            err_request = deepcopy(request_obj)
            err_request.text_answer = 'Invalid arguments'
            self._put(err_request, request_obj.identifier)
            self._put(RequestObject(request_obj.identifier, request_obj.user, termination=True), request_obj.identifier)

    def _on_quiet(self, request_obj, termination_request):
        termination_request.priority = 0
        self._put(termination_request, request_obj.identifier)
        self.sio().emit('interrupt', Timer().now(), to=request_obj.socket_id)

    def _on_sleep(self, request_obj, termination_request):
        self.frozen = True

        ack = deepcopy(request_obj)
        ack.text_answer = 'Sleeping...'

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)

    def _on_memory_wipe(self, request_obj, termination_request):
        self.clear_context(request_obj.preprompt)

        ack = deepcopy(request_obj)
        ack.text_answer = 'Memory wiped.'

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)

    # def _zero_shot_classify(self, text):
    #     t0 = time()
    #     text_embed = get_embedding(text, engine=self._model)
    #     scores = [cosine_similarity(text_embed, label_embed) for label_embed in self._labels_embeddings]
    #     best_label_idx = np.argmax(scores)
    #     ProjectLogger().info(f'{text} --> {self._labels[best_label_idx]} : {scores[best_label_idx]:.3f}')
    #
    #     if best_label_idx != len(self._labels) - 1 and best_label_idx >= self._classif_threshold:
    #         self._dispatch(best_label_idx)
    #     else:
    #         self._dispatch(None)
    #
    #     ProjectLogger().debug(f'{self.__class__.__name__} {time() - t0:.3f} COMMAND exec. time')

    # def run(self):
    #     with ThreadPoolExecutor(max_workers=4) as executor:
    #         while self.running:
    #             try:
    #                 request_obj = self._consume()
    #                 future = executor.submit(self._zero_shot_classify, request_obj.text_request)
    #             except queue.Empty:
    #                 continue
    #
    #     ProjectLogger().info('Command Detector stopped.')
