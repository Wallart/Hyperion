from time import time
from copy import deepcopy
from threading import Lock
from tinydb import TinyDB, Query
from utils.logger import ProjectLogger
from utils.request import RequestObject
from utils.threading import Consumer, Producer
from concurrent.futures import ThreadPoolExecutor
from utils.external_resources_parsing import fetch_urls
from analysis import MAX_TOKENS, acquire_mutex, get_model_token_specs

import os
import queue
import random
import openai
import tiktoken
import numpy as np

CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-4']


class ChatGPT(Consumer, Producer):

    def __init__(self, name, model, no_memory, clear, prompt='base', cache_dir='~/.hyperion'):
        super().__init__()
        self.frozen = False
        ProjectLogger().info(f'{name} using {model} as chat backend. No memory -> {no_memory}')

        self._mutex = Lock()
        self._model = model
        self._botname = name
        self._no_memory = no_memory
        # Seems that we have to reserve some tokens for chat completion...
        self._max_ctx_tokens = int(MAX_TOKENS - (MAX_TOKENS * .05))
        # self._max_ctx_tokens = MAX_TOKENS

        root_dir = os.path.dirname(os.path.dirname(__file__))
        self._resources_dir = os.path.join(root_dir, 'resources')
        self._cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

        self._tokenizer = tiktoken.encoding_for_model(self._model)
        self._tokens_per_message, self._tokens_per_name = get_model_token_specs(self._model)

        db_path = os.path.join(self._cache_dir, 'prompts_db.json')
        if clear and os.path.exists(db_path):
            ProjectLogger().info('Cleared persistent memory.')
            os.remove(db_path)

        self._db = TinyDB(db_path)

        sentences_path = os.path.join(self._resources_dir, 'default_sentences')
        self._deaf_sentences = self._load_file(os.path.join(sentences_path, 'deaf'))
        self._error_sentences = self._load_file(os.path.join(sentences_path, 'dead'))
        self._global_context = self._load_prompt(prompt)

        openai.api_key = self._load_api_key()
        # self._working_memory = []

    @staticmethod
    def _load_file(path):
        with open(path) as f:
            content = f.readlines()
        return [l.strip() for l in content]

    def _load_api_key(self):
        return ChatGPT._load_file(os.path.join(self._resources_dir, 'openai_api_key.txt'))[0]

    def _load_prompt(self, prompt_file):
        content = ChatGPT._load_file(os.path.join(self._resources_dir, 'prompts', prompt_file))

        context = []
        for line in content:
            sp_line = line.split('::')
            role, name, message = sp_line if len(sp_line) == 3 else (sp_line[0], None, sp_line[1])
            message = self._customize_message(message)
            context.append(ChatGPT._build_context_line(role, message, name=name))
        return context

    def _customize_message(self, message):
        return message.replace('{name}', self._botname)

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

    @staticmethod
    def _build_context_line(role, content, name=None):
        if name is None:
            return {'role': role, 'content': content}
        return {'role': role, 'content': content, 'name': name}

    @acquire_mutex
    def _add_to_context(self, new_message):
        cache = [new_message]
        if not self._no_memory:
            cache = self._db.all() + cache
            self._db.insert(new_message)

        while True:
            messages = self._global_context + cache
            found_tokens = self._tokens_count(messages)
            if found_tokens < self._max_ctx_tokens:
                ProjectLogger().info(f'Sending a {found_tokens} tokens request.')
                break
            cache.pop(0)

        return messages

    @acquire_mutex
    def clear_context(self):
        self._db.truncate()

    def answer(self, chat_input, role='user', name=None, stream=True):
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._add_to_context(ChatGPT._build_context_line(role, chat_input, name=name)),
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
            chunked_response = self.answer(request_obj.text_request, name=request_obj.user)
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

            self._add_to_context(ChatGPT._build_context_line('assistant', memory))

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

                    future = executor.submit(self._process_request, request_obj)
                except queue.Empty:
                    continue

        ProjectLogger().info('ChatGPT stopped.')


if __name__ == '__main__':
    chat = ChatGPT('TOTO', 'gpt-3.5-turbo', False, False)
    chat.answer('En quelle année est né Nicolas Sarkozy ?', stream=False)
