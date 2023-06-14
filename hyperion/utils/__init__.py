from pathlib import Path
from hyperion.utils.logger import ProjectLogger

import os
import torch


def get_ctx(args):
    devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
    if torch.cuda.is_available():
        if len(devices_id) == 0:
            devices_id = list(range(torch.cuda.device_count()))

        ctx = [torch.device(f'cuda:{i}') for i in devices_id if i >= 0]
        ctx = ctx if len(ctx) > 0 else [torch.device('cpu')]
    else:
        ProjectLogger().warning('Cannot access GPU.')
        ctx = [torch.device('cpu')]

    ProjectLogger().info('Used context: {}'.format(', '.join([str(x) for x in ctx])))
    return ctx


def load_file(path):
    with open(path) as f:
        content = f.readlines()
    return [l.strip() for l in content]


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProjectPaths(metaclass=Singleton):

    def __init__(self, cache_dir='~/.hyperion'):
        self._init_cache_dir(cache_dir)
        self._init_resources_dir()
        self.pid_dir = Path('/') / 'tmp'
        self.log_dir = Path('/') / 'tmp' / 'log'
        os.makedirs(self.log_dir, exist_ok=True)

    def _init_cache_dir(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir = self.cache_dir.expanduser()

        os.makedirs(self.cache_dir, exist_ok=True)

    def _init_resources_dir(self):
        root_dir = Path(__file__).parents[2]
        self.resources_dir = root_dir / 'resources'
        # in production mode
        if not self.resources_dir.is_dir():
            self.resources_dir = self.cache_dir / 'resources'
            if not self.resources_dir:
                os.makedirs(self.resources_dir, exist_ok=True)
