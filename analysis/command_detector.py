from enum import Enum
from time import time
from utils.logger import ProjectLogger
from utils.threading import Consumer, Producer
from concurrent.futures import ThreadPoolExecutor
from openai.embeddings_utils import cosine_similarity, get_embedding, get_embeddings

import os
import json
import queue
import numpy as np


class ACTIONS(Enum):
    SLEEP = 0
    WAKE = 1
    WIPE = 2
    QUIET = 3


class CommandDetector(Consumer, Producer):
    def __init__(self, threshold=.8):
        super().__init__()

        root_dir = os.path.dirname(os.path.dirname(__file__))
        self._commands_file = os.path.join(root_dir, 'resources', 'default_sentences', 'commands.json')

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

                if action is not None:
                    ProjectLogger().info(f'Command found in "{analyzed_text}"')
                self._dispatch(action)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} COMMAND exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Command Detector stopped.')

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
