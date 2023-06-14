from unidecode import unidecode
from hyperion.utils import ProjectPaths
from hyperion.utils.logger import ProjectLogger

import re
import json

with open(ProjectPaths().resources_dir / 'gpt_models.json') as f:
    CHAT_MODELS = json.load(f)


def sanitize_username(string):
    """
    GPT API only accept username matching this pattern
    :param string:
    :return:
    """
    string = unidecode(string)
    pattern = r"[a-zA-Z0-9_-]{1,64}"
    match = re.search(pattern, string.strip())
    if match:
        return match.group(0)
    else:
        return None


def build_context_line(role, content, name=None):
    if name is None:
        return {'role': role, 'content': content}
    return {'role': role, 'content': content, 'name': name}


def acquire_mutex(fn):
    def wrapper(*args, **kwargs):
        mutex = args[0]._mutex
        try:
            mutex.acquire()
            res = fn(*args, **kwargs)
        finally:
            mutex.release()
        return res
    return wrapper


def get_model_token_specs(model):
    if model.startswith('gpt-3.5-turbo') and model != 'gpt-3.5-turbo-0301':
        ProjectLogger().debug('gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.')
        return get_model_token_specs('gpt-3.5-turbo-0301')
    elif model.startswith('gpt-4') and model != 'gpt-4-0314':
        ProjectLogger().debug('gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.')
        return get_model_token_specs('gpt-4-0314')
    elif model == 'gpt-3.5-turbo-0301':
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == 'gpt-4-0314':
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f'num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.')

    return tokens_per_message, tokens_per_name
