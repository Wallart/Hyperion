from time import time
from openai import OpenAI
from threading import Lock
from openai._types import NOT_GIVEN
from hyperion.utils import load_file
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.request import RequestObject
from concurrent.futures import ThreadPoolExecutor
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.threading import Consumer, Producer
from hyperion.analysis.prompt_manager import PromptManager
from hyperion.utils.external_resources_parsing import fetch_urls
from hyperion.analysis import CHAT_MODELS, acquire_mutex, get_model_token_specs, build_context_line, sanitize_username

import os
import queue
import random
import tiktoken
import numpy as np


class ChatGPT(Consumer, Producer):

    def __init__(self, name, model, no_memory, clear, prompt='base'):
        super().__init__()
        ProjectLogger().info(f'{name} using {model} as chat backend. No memory -> {no_memory}')

        self._mutex = Lock()
        self._model = model
        self._no_memory = no_memory
        self._tokenizer = tiktoken.encoding_for_model(self._model)

        self.prompt_manager = PromptManager(name, prompt, clear)

        sentences_path = ProjectPaths().resources_dir / 'default_sentences'
        self._deaf_sentences = load_file(sentences_path / 'deaf')
        self._error_sentences = load_file(sentences_path / 'dead')
        self._memory_sentences = load_file(sentences_path / 'memory')

        openai_api = ProjectPaths().resources_dir / 'keys' / 'openai_api.key'
        self._client = OpenAI(api_key=os.environ['OPENAI_API'] if 'OPENAI_API' in os.environ else load_file(openai_api)[0])

        self._video_ctx = None
        self._video_ctx_timestamp = time()

    @staticmethod
    def max_tokens(model):
        max_tokens = CHAT_MODELS[model]
        # Seems that we have to reserve some tokens for chat completion...
        return int(max_tokens - (max_tokens * .05))

    def get_model(self):
        return self._model

    def set_model(self, model):
        if model not in CHAT_MODELS.keys():
            return False
        self._model = model
        return True

    def _tokens_count(self, messages, llm=None):
        """
        Returns the number of tokens used by a list of messages.
        See https://platform.openai.com/docs/guides/vision for images token count.
        """
        tokens_per_message, tokens_per_name = get_model_token_specs(self._model if llm is None else llm)

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if type(value) == str:
                    num_tokens += len(self._tokenizer.encode(value))
                elif value[0]['type'] == 'text':
                    num_tokens += len(self._tokenizer.encode(value[0]['text']))
                elif value[0]['image_url']['detail'] == 'low':
                    num_tokens += 85
                elif value[0]['image_url']['detail'] == 'high':
                    raise NotImplementedError('Depend of image size and may vary.')

                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @acquire_mutex
    def _add_to_context(self, new_message, preprompt=None, llm=None):
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

        dropped_messages = False
        while True:
            messages = self.prompt_manager.preprompt(preprompt) + cache
            found_tokens = self._tokens_count(messages, llm)
            if found_tokens < ChatGPT.max_tokens(self._model if llm is None else llm):
                ProjectLogger().info(f'Sending a {found_tokens} tokens request.')
                break
            cache.pop(0)
            dropped_messages = True

        return messages, dropped_messages

    @acquire_mutex
    def clear_context(self, preprompt=None):
        self.prompt_manager.truncate(preprompt)
        ProjectLogger().warning(f'Memory wiped for {preprompt}.')

    def add_video_context(self, frame_description):
        if self._video_ctx is None or frame_description != self._video_ctx:
            self._video_ctx = frame_description
            self._video_ctx_timestamp = time()

    @acquire_mutex
    def add_document_context(self, pages, preprompt):
        for page in pages:
            messages = [build_context_line('system', s.strip()) for s in page.split('.')]
            _ = [self.prompt_manager.insert(message, preprompt) for message in messages]

    def answer(self, chat_input, role='user', name=None, preprompt=None, llm=None, stream=True):
        if name is not None:
            name = sanitize_username(name)

        messages, dropped_messages = self._add_to_context(build_context_line(role, chat_input, name=name), preprompt, llm)
        # TODO vision-preview workaround for cutted sentences
        max_tokens = 4096 if 'vision-preview' in llm else NOT_GIVEN
        response = self._client.chat.completions.create(model=self._model if llm is None else llm, messages=messages, stream=stream, max_tokens=max_tokens)
        return response, dropped_messages

    def _dispatch_sentence(self, sentence, sentence_num, t0, request_obj):
        sentence = sentence.strip()

        new_request_obj = RequestObject.copy(request_obj)
        new_request_obj.text_answer = sentence
        new_request_obj.num_answer = sentence_num
        new_request_obj.timestamp = t0
        self._dispatch(new_request_obj)
        ProjectLogger().info(f'ChatGPT : {sentence}')

    def _dispatch_memory_warning(self, request_obj, sentence_num=None, randomized=False):
        # in randomize mode dispatch warning only 1 time out 3
        if randomized and random.choices(range(10), weights=[1] * 10) != 9:
            return False

        placeholder = self._memory_sentences[random.randint(0, len(self._memory_sentences) - 1)]

        new_request_obj = RequestObject.copy(request_obj)
        new_request_obj.text_answer = placeholder
        if sentence_num is not None:
            new_request_obj.num_answer = sentence_num
        self._dispatch(new_request_obj)
        return True

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
            ProjectLogger().info('Requesting ChatGPT...')
            ProjectLogger().info(f'{request_obj.user} : {request_obj.text_request}')

            input_text = fetch_urls(request_obj.text_request)
            answer_opts = dict(name=request_obj.user, preprompt=request_obj.preprompt, llm=request_obj.llm)
            chunked_response, dropped_messages = self.answer(input_text, **answer_opts)
            ProjectLogger().info(f'ChatGPT answered in {time() - t0:.3f} sec(s)')
            if dropped_messages and self._dispatch_memory_warning(request_obj, sentence_num, randomized=True):
                sentence_num += 1

            for chunk in chunked_response:
                finish_reason = chunk.choices[0].finish_reason
                # TODO vision-preview support
                if 'finish_details' in chunk.choices[0].model_extra and chunk.choices[0].model_extra['finish_details'] is not None:
                    finish_reason = chunk.choices[0].model_extra['finish_details']['type']

                if finish_reason == 'stop':
                    if sentence != '':
                        self._dispatch_sentence(sentence, sentence_num, t0, request_obj)
                    break
                elif finish_reason == 'length':
                    ProjectLogger().warning('Not enough left tokens to generate a complete answer')
                    self._dispatch_memory_warning(request_obj, sentence_num)
                    break
                elif finish_reason is not None:
                    ProjectLogger().warning('Unsupported finish reason')
                    self._dispatch_error(sentence_num, request_obj)

                answer = chunk.choices[0].delta
                if hasattr(answer, 'content') and answer.content is not None:
                    content = answer.content
                    sentence += content
                    memory += content

                    sentence_ends = [sentence.find(e) for e in ['. ', '! ', '? ', '; ']]
                    sentence_end = sentence_ends[np.argmax(sentence_ends)] + 1
                    # if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                    if sentence_end > 0:
                        self._dispatch_sentence(sentence[:sentence_end], sentence_num, t0, request_obj)
                        sentence = sentence[sentence_end:]
                        sentence_num += 1

            _ = self._add_to_context(build_context_line('assistant', memory), request_obj.preprompt, request_obj.llm)

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

                    ack = RequestObject.copy(request_obj)
                    ack.text_answer = '<ACK>'
                    ack.silent = True
                    self._dispatch(ack)

                    if request_obj.text_request == '':
                        ack = RequestObject.copy(request_obj)
                        ack.text_answer = '<CONFUSED>'
                        ack.silent = True
                        ack.priority = 0
                        self._dispatch(ack)

                        # 1 in 10 chance of receiving a notification that the message wasn't heard.
                        request_obj.text_answer = ''
                        if random.choices(range(10), weights=[1] * 10) != 9:
                            placeholder = self._deaf_sentences[random.randint(0, len(self._deaf_sentences) - 1)]
                            request_obj.text_answer = placeholder
                            request_obj.num_answer += 1

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
