# from openai api
MAX_TOKENS = 4097


def acquire_mutex(fn):
    def wrapper(*args):
        mutex = args[0]._mutex
        try:
            mutex.acquire()
            res = fn(*args)
        finally:
            mutex.release()
        return res
    return wrapper
