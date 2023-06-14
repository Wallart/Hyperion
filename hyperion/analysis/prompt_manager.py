from datetime import datetime
from tinydb import TinyDB, Query
from hyperion.utils import load_file
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.paths import ProjectPaths
from hyperion.analysis import build_context_line, sanitize_username


class PromptManager:

    def __init__(self, bot_name, initial_preprompt_name, clear=False):
        self._bot_name = bot_name
        self._db = {}
        self._preprompt = {}

        self._clear = clear
        self._current_preprompt_name = initial_preprompt_name

        self._fetch_db(initial_preprompt_name)
        self._fetch_preprompt(initial_preprompt_name)

    def _fetch_db(self, preprompt):
        db_path = ProjectPaths().cache_dir / f'prompts_db_{preprompt}.json'
        if self._clear and db_path.exists():
            ProjectLogger().info('Cleared persistent memory.')
            db_path.unlink()

        self._db[preprompt] = TinyDB(db_path)

    def _fetch_preprompt(self, preprompt_name):
        content = load_file(ProjectPaths().resources_dir / 'prompts' / preprompt_name)

        prompt_lines = []
        start_tokens = ['system::', 'user::', 'assistant::']
        for line in content:
            if True in [line.startswith(t) for t in start_tokens]:
                prompt_lines.append(line)
            elif len(line.strip()) > 0:
                prompt_lines[-1] = prompt_lines[-1].strip() + ' ' + line

        context = []
        for prompt_line in prompt_lines:
            sp_line = prompt_line.split('::')
            role, name, message = sp_line if len(sp_line) == 3 else (sp_line[0], None, sp_line[1])
            if name is not None:
                name = sanitize_username(name)
            context.append(build_context_line(role, message, name=name))

        self._preprompt[preprompt_name] = context

    def _customize_preprompt(self, message):
        return message.replace('{name}', self._bot_name).replace('{date}', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    def _get_db(self, preprompt_name):
        preprompt_name = self._current_preprompt_name if preprompt_name is None else preprompt_name
        if preprompt_name not in self._db:
            self._fetch_db(preprompt_name)
        return self._db[preprompt_name]

    def _get_preprompt(self, preprompt_name):
        preprompt_name = self._current_preprompt_name if preprompt_name is None else preprompt_name
        if preprompt_name not in self._preprompt:
            self._fetch_preprompt(preprompt_name)

        processed_preprompt = []
        for line in self._preprompt[preprompt_name]:
            newline = line.copy()
            newline['content'] = self._customize_preprompt(newline['content'])
            processed_preprompt.append(newline)

        return processed_preprompt

    @staticmethod
    def list_prompts():
        prompts = [p.stem for p in (ProjectPaths().resources_dir / 'prompts').glob('*')]
        return prompts

    def get_prompt(self):
        return self._current_preprompt_name

    def set_prompt(self, prompt_name):
        if prompt_name not in PromptManager.list_prompts():
            return False
        self._current_preprompt_name = prompt_name
        return True

    def all(self, preprompt_name=None):
        return self._get_db(preprompt_name).all()

    def preprompt(self, preprompt_name=None):
        return self._get_preprompt(preprompt_name)

    def insert(self, new_message, preprompt_name=None):
        self._get_db(preprompt_name).insert(new_message)

    def truncate(self, preprompt_name=None):
        self._get_db(preprompt_name).truncate()
