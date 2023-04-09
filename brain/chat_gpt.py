from time import time
from threading import Lock
from tinydb import TinyDB, Query
from utils.logger import ProjectLogger
from brain import MAX_TOKENS, acquire_mutex
from transformers import GPT2TokenizerFast
from utils.threading import Consumer, Producer
from concurrent.futures import ThreadPoolExecutor

import os
import random
import openai

CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4']


class ChatGPT(Consumer, Producer):

    def __init__(self, name, model, no_memory, cache_dir='~/.hyperion'):
        super().__init__()

        ProjectLogger().info(f'{name} using {model} as chat backend. No memory -> {no_memory}')

        self._mutex = Lock()
        self._model = model
        self._botname = name
        self._no_memory = no_memory
        # 5% less than max tokens because we don't know exactly what are tokens.
        # Usually they are words, sometimes it's just a letter or a comma.
        self._max_ctx_tokens = MAX_TOKENS - (MAX_TOKENS * .05)

        root_dir = os.path.dirname(os.path.dirname(__file__))
        self._resources_dir = os.path.join(root_dir, 'resources')
        self._cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir=os.path.join(self._cache_dir, 'tokenizer'))
        self._db = TinyDB(os.path.join(self._cache_dir, 'prompts_db.json'))

        sentences_path = os.path.join(self._resources_dir, 'default_sentences')
        self._deaf_sentences = self._load_file(os.path.join(sentences_path, 'deaf'))
        self._error_sentences = self._load_file(os.path.join(sentences_path, 'dead'))
        self._global_context = self._load_prompt('base')

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
            cache += self._db.all()
            self._db.insert(new_message)

        while True:
            messages = self._global_context + cache
            if self._tokens_count(messages) <= self._max_ctx_tokens:
                break
            cache.pop()

        return messages

    @acquire_mutex
    def _clear_context(self):
        raise NotImplemented()

    def answer(self, input, role='user', stream=True):
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._add_to_context(ChatGPT._build_context_line(role, input)),
            stream=stream
        )
        return response

    def _process_request(self, request):
        try:
            t0 = time()
            ProjectLogger().info(f'Requesting ChatGPT...')
            chunked_response = self.answer(request)
            ProjectLogger().info(f'ChatGPT answered in {time() - t0:.3f} sec(s)')

            memory = ''
            sentence = ''
            for chunk in chunked_response:
                if chunk['choices'][0]['finish_reason'] == 'stop':
                    break

                anwser = chunk['choices'][0]['delta']
                if 'content' in anwser:
                    content = anwser['content']
                    sentence += content
                    memory += content
                    if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                        sentence = sentence.strip()
                        self._dispatch(sentence)
                        ProjectLogger().info(f'ChatGPT : {sentence}')
                        sentence = ''

            self._add_to_context(ChatGPT._build_context_line('assistant', memory))

        except Exception as e:
            ProjectLogger().error(f'ChatGPT had a stroke. {e}')
            self._dispatch(self._error_sentences[random.randint(0, len(self._error_sentences) - 1)])
            # TODO Make it thread safe
            # self._clear_context()

        # To close streaming response
        self._dispatch(None)

    def run(self) -> None:
        # while True:
        #     request = self._in_queue.get()
        #     if request is None:
        #         self._dispatch(self._deaf_sentences[random.randint(0, len(self._deaf_sentences) - 1)])
        #         self._dispatch(None)  # To close streaming response
        #         continue
        #
        #     self.process_request(request)

        with ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                request = self._in_queue.get()
                if request is None:
                    self._dispatch(self._deaf_sentences[random.randint(0, len(self._deaf_sentences) - 1)])
                    self._dispatch(None)  # To close streaming response
                    continue

                future = executor.submit(self._process_request, request)


if __name__ == '__main__':
    chat = ChatGPT()
    chat.answer('En quelle année est né Nicolas Sarkozy ?', stream=False)
