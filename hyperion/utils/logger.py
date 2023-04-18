from hyperion.utils import Singleton, ProjectPaths

import os
import sys
import logging


class ProjectLogger(metaclass=Singleton):
    def __init__(self, *args):
        conf = {}
        level = logging.INFO
        self._name = os.path.basename(sys.argv[0])
        if len(args) > 0:
            opts, name = args
            level = logging.DEBUG if opts.debug else level
            self._name = name
            self._foreground = not opts.daemon if hasattr(opts, 'daemon') else True

            if not self._foreground:
                conf['filename'] = ProjectPaths().log_dir / f'{self._name}.log'

        logging.basicConfig(**conf)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(level)

    def error(self, msg):
        self._logger.log(logging.ERROR, msg)

    def warning(self, msg):
        self._logger.log(logging.WARNING, msg)

    def debug(self, msg):
        self._logger.log(logging.DEBUG, msg)

    def info(self, msg):
        self._logger.log(logging.INFO, msg)
