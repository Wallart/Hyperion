import openai

from utils.threading import Consumer, Producer


class ChatGPT():

    def __init__(self, max_memory=400):
        self._max_memory = max_memory
        self._model = 'gpt-3.5-turbo'
        with open('./resources/api_key.txt') as f:
            api_key = f.readlines()[0]

        openai.api_key = api_key
        self._global_context = [
            ChatGPT._build_context_line('system', 'Tu es un assistant virtuel vraiment sympatique et plein d\'humour nommé Hypérion.'),
            ChatGPT._build_context_line('user', 'Bonjour Hypérion comment vas-tu aujourd\'hui ?'),
            ChatGPT._build_context_line('assistant', 'Je vais bien Julien ! Merci de demander.'),
            ChatGPT._build_context_line('user', 'Parfait, toi et moi nous allons discuter de plein de sujets différents.')
        ]
        self._working_memory = []
    @staticmethod
    def _build_context_line(role, content):
        return {'role': role, 'content': content}

    def answer(self, input):
        new_message = ChatGPT._build_context_line('user', input)
        self._working_memory.append(new_message)

        messages = self._global_context + self._working_memory
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=messages
        )
        parsed_response = dict(response['choices'][0]['message'])
        self._working_memory.append(parsed_response)
        if len(self._working_memory) > self._max_memory:
            self._working_memory.pop()

        return parsed_response['content']


if __name__ == '__main__':
    chat = ChatGPT()
    chat.answer('En quelle année est né Nicolas Sarkozy ?')
