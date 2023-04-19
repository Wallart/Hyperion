from hyperion.utils.logger import ProjectLogger

# from openai api
MAX_TOKENS = {
    'gpt-3.5-turbo': 4096,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}


def load_file(path):
    with open(path) as f:
        content = f.readlines()
    return [l.strip() for l in content]


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
    if model == 'gpt-3.5-turbo':
        ProjectLogger().debug('gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.')
        return get_model_token_specs('gpt-3.5-turbo-0301')
    elif model == 'gpt-4' or model == 'gpt-4-32k':
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
