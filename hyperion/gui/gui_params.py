from hyperion.utils import Singleton, ProjectPaths


import json


class GUIParams(dict, metaclass=Singleton):

    def __init__(self):
        super().__init__()

        # self._gui_params = {}
        self._savefile = ProjectPaths().cache_dir / 'gui_params.json'
        if self._savefile.exists():
            with open(self._savefile) as f:
                for k, v in json.load(f).items():
                    self[k] = v

    def save(self):
        with open(self._savefile, 'w') as f:
            f.write(json.dumps(self))
