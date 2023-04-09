from time import time
from utils.threading import Consumer, Producer

import os
import random
import openai
import logging

# from openai api
MAX_TOKENS = 4097


class ChatGPT(Consumer, Producer):

    def __init__(self, name='Hypérion', model='gpt-3.5-turbo'):
        super().__init__()

        self._error_sentences = [
            'Désolé j\'ai le cerveau lent.',
            'Je crois que je fais un micro-AVC...',
            'Désolé ! Je sais pas, je sais plus... Je suis fatigué.',
            'Qu\'est-ce qui est jaune et qui attend ? Jonathan !'
        ]

        self._name = name
        self._model = model

        # 5% less than max tokens because we don't know exactly what are tokens.
        # Usually they are words, sometimes it's just a letter or a comma.
        self._max_ctx_tokens = MAX_TOKENS - (MAX_TOKENS * .05)

        with open(os.path.join(os.getcwd(), 'resources', 'openai_api_key.txt')) as f:
            api_key = f.readlines()[0]

        openai.api_key = api_key
        self._global_context = [
            ChatGPT._build_context_line('system', f'Tu es un assistant virtuel vraiment sympatique et plein d\'humour nommé {self._name}, qui répond toujours en français. Tu parles avec plusieurs personnes, mais refuses de répondre aux inconnus.'),
            ChatGPT._build_context_line('user', f'Julien : Bonjour {self._name} comment vas-tu aujourd\'hui ?'),
            ChatGPT._build_context_line('assistant', 'Je vais bien Julien ! Merci de demander.'),
            ChatGPT._build_context_line('user', 'Julien : Mon chat s\'appelle Petit Poulet.'),
            ChatGPT._build_context_line('assistant', 'C\'est noté.'),
            ChatGPT._build_context_line('user', f'Michel : Salut {self._name} !'),
            ChatGPT._build_context_line('assistant', 'Bonjour Michel.'),
            ChatGPT._build_context_line('user', f'Unknown : Bonjour {self._name} !'),
            ChatGPT._build_context_line('assistant', 'Désolé ma maman m\'a dit de ne pas répondre aux inconnus.')
        ]
        self._working_memory = []

    def tokens_count(self):
        count = 0
        for line in self._global_context + self._working_memory:
            count += len(line['content'].split(' '))
        return count

    @staticmethod
    def _build_context_line(role, content):
        return {'role': role, 'content': content}

    def answer(self, input, role='user', stream=True):
        new_message = ChatGPT._build_context_line(role, input)
        self._working_memory.append(new_message)

        while self.tokens_count() > self._max_ctx_tokens:
            self._working_memory.pop()

        messages = self._global_context + self._working_memory
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=messages,
            stream=stream
        )
        return response

    def run(self) -> None:
        while True:
            request = self._in_queue.get()
            t0 = time()

            try:
                logging.info(f'Requesting ChatGPT...')
                chunked_response = self.answer(request)
                logging.info(f'ChatGPT answered in {time() - t0:.3f} sec(s)')

                memory = ''
                sentence = ''
                for chunk in chunked_response:
                    stopped = chunk['choices'][0]['finish_reason'] == 'stop'
                    if stopped:
                        break

                    anwser = chunk['choices'][0]['delta']
                    if 'content' in anwser:
                        content = anwser['content']
                        sentence += content
                        memory += content
                        if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                            sentence = sentence.strip()
                            self._dispatch(sentence)
                            sentence = ''

                memory = ChatGPT._build_context_line('assistant', memory)
                self._working_memory.append(memory)

            except Exception as e:
                logging.error(f'ChatGPT had a stroke. {e}')
                logging.warning(f'Wiping working memory.')
                self._working_memory = []
                self._dispatch(self._error_sentences[random.randint(0, len(self._error_sentences) - 1)])

            # To close streaming response
            self._dispatch(None)


if __name__ == '__main__':
    chat = ChatGPT()
    chat.answer('En quelle année est né Nicolas Sarkozy ?', stream=False)
