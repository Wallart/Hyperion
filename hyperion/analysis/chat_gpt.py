from time import time
from copy import deepcopy
from threading import Lock
from hyperion.utils import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.request import RequestObject
from concurrent.futures import ThreadPoolExecutor
from hyperion.utils.threading import Consumer, Producer
from hyperion.analysis.prompt_manager import PromptManager
from hyperion.utils.external_resources_parsing import fetch_urls
from hyperion.analysis import MAX_TOKENS, acquire_mutex, get_model_token_specs, load_file, build_context_line

import queue
import random
import openai
import tiktoken
import numpy as np

CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-4']


class ChatGPT(Consumer, Producer):

    def __init__(self, name, model, no_memory, clear, prompt='base'):
        super().__init__()
        self.frozen = False
        ProjectLogger().info(f'{name} using {model} as chat backend. No memory -> {no_memory}')

        self._mutex = Lock()
        self._model = model
        self._no_memory = no_memory
        # Seems that we have to reserve some tokens for chat completion...
        self._max_ctx_tokens = int(MAX_TOKENS - (MAX_TOKENS * .05))
        # self._max_ctx_tokens = MAX_TOKENS

        self._tokenizer = tiktoken.encoding_for_model(self._model)
        self._tokens_per_message, self._tokens_per_name = get_model_token_specs(self._model)

        self.prompt_manager = PromptManager(name, prompt, clear)

        sentences_path = ProjectPaths().resources_dir / 'default_sentences'
        self._deaf_sentences = load_file(sentences_path / 'deaf')
        self._error_sentences = load_file(sentences_path / 'dead')

        openai.api_key = load_file(ProjectPaths().resources_dir / 'keys' / 'openai_api.key')[0]

        self._video_ctx = None
        self._video_ctx_timestamp = time()

    def _tokens_count(self, messages):
        """Returns the number of tokens used by a list of messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += self._tokens_per_message
            for key, value in message.items():
                num_tokens += len(self._tokenizer.encode(value))
                if key == 'name':
                    num_tokens += self._tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @acquire_mutex
    def _add_to_context(self, new_message, preprompt=None):
        cache = [new_message]
        if self._video_ctx is not None and time() - self._video_ctx_timestamp < 20:
            video_ctx = f'[VIDEO STREAM] {self._video_ctx}'
            cache.insert(0, build_context_line('system', video_ctx))
            ProjectLogger().info(video_ctx)
        else:
            self._video_ctx = None

        if not self._no_memory:
            cache = self.prompt_manager.all(preprompt) + cache
            self.prompt_manager.insert(new_message, preprompt)

        while True:
            messages = self.prompt_manager.preprompt(preprompt) + cache
            found_tokens = self._tokens_count(messages)
            if found_tokens < self._max_ctx_tokens:
                ProjectLogger().info(f'Sending a {found_tokens} tokens request.')
                break
            cache.pop(0)

        return messages

    @acquire_mutex
    def clear_context(self, preprompt=None):
        self.prompt_manager.truncate(preprompt)

    def add_video_context(self, frame_description):
        if self._video_ctx is None or frame_description != self._video_ctx:
            self._video_ctx = frame_description
            self._video_ctx_timestamp = time()

    def answer(self, chat_input, role='user', name=None, preprompt=None, stream=True):
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._add_to_context(build_context_line(role, chat_input, name=name), preprompt),
            stream=stream
        )
        return response

    def _dispatch_sentence(self, sentence, sentence_num, t0, request_obj):
        sentence = sentence.strip()

        new_request_obj = deepcopy(request_obj)
        new_request_obj.text_answer = sentence
        new_request_obj.num_answer = sentence_num
        new_request_obj.timestamp = t0
        self._dispatch(new_request_obj)
        ProjectLogger().info(f'ChatGPT : {sentence}')

    def _dispatch_error(self, sentence_num, request_obj):
        placeholder = self._error_sentences[random.randint(0, len(self._error_sentences) - 1)]
        request_obj.text_answer = placeholder
        request_obj.num_answer = sentence_num
        self._dispatch(request_obj)

    def _process_request(self, request_obj):

        t0 = time()
        memory = ''
        sentence = ''
        sentence_num = 0
        try:
            request_obj.text_request = fetch_urls(request_obj.text_request)
            ProjectLogger().info('Requesting ChatGPT...')
            ProjectLogger().info(f'{request_obj.user} : {request_obj.text_request}')
            chunked_response = self.answer(request_obj.text_request, name=request_obj.user, preprompt=request_obj.preprompt)
            ProjectLogger().info(f'ChatGPT answered in {time() - t0:.3f} sec(s)')

            for chunk in chunked_response:
                if chunk['choices'][0]['finish_reason'] == 'stop':
                    if sentence != '':
                        self._dispatch_sentence(sentence, sentence_num, t0, request_obj)
                    break

                if chunk['choices'][0]['finish_reason'] == 'length':
                    ProjectLogger().warning('Not enough left tokens to generate a complete answer')
                    self._dispatch_error(sentence_num, request_obj)
                    break

                answer = chunk['choices'][0]['delta']
                if 'content' in answer:
                    content = answer['content']
                    sentence += content
                    memory += content

                    sentence_ends = [sentence.find(e) for e in ['. ', '! ', '? ', '.\n', '!\n', '?\n']]
                    sentence_end = sentence_ends[np.argmax(sentence_ends)] + 1
                    # if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                    if sentence_end > 0:
                        self._dispatch_sentence(sentence[:sentence_end], sentence_num, t0, request_obj)
                        sentence = sentence[sentence_end:]
                        sentence_num += 1

            self._add_to_context(build_context_line('assistant', memory), request_obj.preprompt)

        except Exception as e:
            ProjectLogger().error(f'ChatGPT had a stroke. {e}')
            self._dispatch_error(sentence_num, request_obj)

        # To close streaming response
        self._dispatch(RequestObject(request_obj.identifier, request_obj.user, termination=True))

    def run(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                try:
                    request_obj = self._consume()
                    if self.frozen:
                        continue

                    if request_obj.text_request == '':
                        placeholder = self._deaf_sentences[random.randint(0, len(self._deaf_sentences) - 1)]
                        request_obj.text_answer = placeholder
                        self._dispatch(request_obj)
                        # To close streaming response
                        self._dispatch(RequestObject(request_obj.identifier, request_obj.user, termination=True))
                        continue

                    _ = executor.submit(self._process_request, request_obj)
                except queue.Empty:
                    continue

        ProjectLogger().info('ChatGPT stopped.')


if __name__ == '__main__':
    chat = ChatGPT('TOTO', 'gpt-3.5-turbo', False, False)
    chat.answer('En quelle année est né Nicolas Sarkozy ?', stream=False)
