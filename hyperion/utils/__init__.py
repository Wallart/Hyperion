from sys import platform

import os


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_pid_root():
    if platform == 'linux' or platform == 'linux2':
        return os.path.join(os.path.sep, 'tmp')
    elif platform == 'darwin':
        return os.path.join(os.path.sep, 'tmp')
    elif platform == 'win32':
        raise NotImplementedError()


def get_log_root():
    if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
        path = os.path.join(os.path.sep, 'tmp', 'log')
        os.makedirs(path, exist_ok=True)
        return path
    elif platform == 'win32':
        raise NotImplementedError()
