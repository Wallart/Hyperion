from time import time
from copy import deepcopy
from threading import Lock
from tinydb import TinyDB, Query
from analysis import MAX_TOKENS, acquire_mutex
from utils.logger import ProjectLogger
from transformers import GPT2TokenizerFast
from utils.request import RequestObject
from utils.threading import Consumer, Producer
from concurrent.futures import ThreadPoolExecutor
from utils.external_resources_parsing import fetch_urls

import os
import queue
import random
import openai

CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4']


class ChatGPT(Consumer, Producer):

    def __init__(self, name, model, no_memory, clear, prompt='base', cache_dir='~/.hyperion'):
        super().__init__()
        self.frozen = False
        ProjectLogger().info(f'{name} using {model} as chat backend. No memory -> {no_memory}')

        self._mutex = Lock()
        self._model = model
        self._botname = name
        self._no_memory = no_memory
        # 5% less than max tokens because we don't know exactly what are tokens.
        # Usually they are words, sometimes it's just a letter or a comma.
        self._max_ctx_tokens = int(MAX_TOKENS - (MAX_TOKENS * .05))

        root_dir = os.path.dirname(os.path.dirname(__file__))
        self._resources_dir = os.path.join(root_dir, 'resources')
        self._cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

        # TODO Replace by openai tiktoken library
        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir=os.path.join(self._cache_dir, 'tokenizer'))
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
            role, message = line.split('::')
            message = self._customize_message(message)
            context.append(ChatGPT._build_context_line(role, message))
        return context

    def _customize_message(self, message):
        return message.replace('{name}', self._botname)

    def _tokens_count(self, ctx):
        return sum([len(self._tokenizer.tokenize(l['content'])) for l in ctx])

    @staticmethod
    def _build_context_line(role, content):
        return {'role': role, 'content': content}

    @acquire_mutex
    def _add_to_context(self, new_message):
        cache = [new_message]
        if not self._no_memory:
            cache = self._db.all() + cache
            self._db.insert(new_message)

        while True:
            messages = self._global_context + cache
            if self._tokens_count(messages) <= self._max_ctx_tokens:
                break
            cache.pop(0)

        return messages

    @acquire_mutex
    def clear_context(self):
        self._db.truncate()

    def answer(self, chat_input, role='user', stream=True):
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._add_to_context(ChatGPT._build_context_line(role, chat_input)),
            stream=stream
        )
        return response

    def _process_request(self, request_obj):

        t0 = time()
        memory = ''
        sentence = ''
        sentence_num = 0
        try:
            request_obj.text_request = fetch_urls(request_obj.text_request)
            ProjectLogger().info('Requesting ChatGPT...')
            chat_input = f'{request_obj.user} : {request_obj.text_request}'
            ProjectLogger().info(f'{chat_input}')
            chunked_response = self.answer(chat_input)
            ProjectLogger().info(f'ChatGPT answered in {time() - t0:.3f} sec(s)')

            for chunk in chunked_response:
                if chunk['choices'][0]['finish_reason'] == 'stop':
                    break

                answer = chunk['choices'][0]['delta']
                if 'content' in answer:
                    content = answer['content']
                    sentence += content
                    memory += content
                    if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                        sentence = sentence.strip()

                        new_request_obj = deepcopy(request_obj)
                        new_request_obj.text_answer = sentence
                        new_request_obj.num_answer = sentence_num
                        new_request_obj.timestamp = t0
                        self._dispatch(new_request_obj)
                        ProjectLogger().info(f'ChatGPT : {sentence}')
                        sentence = ''
                        sentence_num += 1

            self._add_to_context(ChatGPT._build_context_line('assistant', memory))

        except Exception as e:
            ProjectLogger().error(f'ChatGPT had a stroke. {e}')
            placeholder = self._error_sentences[random.randint(0, len(self._error_sentences) - 1)]
            request_obj.text_answer = placeholder
            request_obj.num_answer = sentence_num
            self._dispatch(request_obj)

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
    chat = ChatGPT()
    chat.answer('En quelle année est né Nicolas Sarkozy ?', stream=False)
