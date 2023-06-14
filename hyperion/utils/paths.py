from pathlib import Path
from hyperion.utils.singleton import Singleton

import os


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
