from enum import Enum
from time import time
from googlesearch import search
from hyperion.utils.timer import Timer
from datetime import datetime, timedelta
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager
from concurrent.futures import ThreadPoolExecutor
from hyperion.utils.memory_utils import MANAGER_TOKEN
from hyperion.utils.identity_store import IdentityStore
from hyperion.utils.threading import Consumer, Producer
from hyperion.utils.task_scheduler import TaskScheduler
from hyperion.utils.request import RequestObject, KeepAliveSet
from hyperion.utils.external_resources_parsing import load_url

import re
import json
import shlex
import queue
import argparse


class ACTIONS(Enum):
    DRAW = 0
    QUERY = 1
    SCHEDULE = 2
    WIPE = 3
    QUIET = 4
    SEARCH = 5


class InterpretedCommandDetector(Consumer, Producer):
    def __init__(self, sio_delegate):
        super().__init__()
        self.sio = sio_delegate
        self.chat_delegate = None
        self.img_delegate = None
        self.img_intake = None
        self._commands_file = ProjectPaths().resources_dir / 'default_sentences' / 'interpreted_commands.json'
        self._cmd_buffer = ''
        self._cmd_buffer_timestamp = 0

        with open(self._commands_file) as f:
            self._commands = json.load(f)

        self._memoryManager = BaseManager(('', 5602), bytes(MANAGER_TOKEN, encoding='utf8'))
        self._memoryManager.register('query_index')
        self._memoryManager.connect()

    def set_chat_delegate(self, chat_delegate):
        self.chat_delegate = chat_delegate

    def set_img_delegate(self, img_delegate):
        self.img_delegate = img_delegate
        self.img_intake = img_delegate.get_intake()

    def run(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            while self.running:
                try:
                    request_obj = self._consume()
                    t0 = time()

                    # Flushing commands stucked for more than 30 secs
                    if time() - self._cmd_buffer_timestamp > 30 and self._cmd_buffer != '':
                        ProjectLogger().warning(f'Flushing "{self._cmd_buffer}". Stucked for more than 30 sec(s)')
                        self._cmd_buffer = ''

                    # don't forward termination requests if there is work still pending
                    pending = request_obj.identifier in KeepAliveSet()
                    if request_obj.termination:
                        if pending:
                            KeepAliveSet().add_termination(request_obj.identifier, request_obj)
                        else:
                            self._dispatch(request_obj)

                        continue

                    text_answer = self._cmd_buffer + request_obj.text_answer
                    action, found_cmd, found_pattern = self._search_command(request_obj, text_answer)

                    if action is None and self._cmd_buffer == '':
                        self._dispatch(request_obj)
                    elif action is not None:
                        ProjectLogger().info(f'Command found in "{found_cmd}"')

                        ack = RequestObject.copy(request_obj)
                        ack.text_answer = '<CMD>'
                        ack.silent = True
                        ack.priority = 0
                        self._dispatch(ack)

                        _ = executor.submit(self._process_command, request_obj, action, found_cmd, found_pattern)

                    ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} COMMAND exec. time')
                except queue.Empty:
                    continue

        ProjectLogger().info('Interpreted Command Detector stopped.')

    def _search_command(self, request_obj, text_answer):
        action = None
        found_cmd = None
        found_pattern = None
        for i, regex_pattern in enumerate(self._commands.values()):
            found_pattern = fr'{regex_pattern}'
            res = re.search(found_pattern, text_answer)
            if res is not None:
                found_cmd = res.group()
                action = i
                request_obj.text_answer = text_answer
                self._cmd_buffer = ''
                break

            partial_pattern = regex_pattern.split(' ')[0]
            res = re.search(fr'{partial_pattern}', text_answer)
            comma_count = text_answer.count('"')
            if res is not None and comma_count == 1:
                self._cmd_buffer = text_answer
                self._cmd_buffer_timestamp = time()
                break
            elif res is not None and comma_count == 0:
                # command is ill formed...
                self._cmd_buffer = ''
                break
        return action, found_cmd, found_pattern

    def _process_command(self, request_obj, action, found_cmd, found_pattern):
        if action == ACTIONS.DRAW.value:
            self._argparsed_command(self._on_draw, found_cmd, found_pattern, request_obj)
        elif action == ACTIONS.QUERY.value:
            self._argparsed_command(self._on_query, found_cmd, found_pattern, request_obj)
        elif action == ACTIONS.SCHEDULE.value:
            self._argparsed_command(self._on_schedule, found_cmd, found_pattern, request_obj)
        elif action == ACTIONS.SEARCH.value:
            self._argparsed_command(self._on_search, found_cmd, found_pattern, request_obj)
        elif action == ACTIONS.QUIET.value:
            self._on_quiet(request_obj)
        elif action == ACTIONS.WIPE.value:
            self._on_memory_wipe(request_obj)

    def _on_schedule(self, command_line, regex_pattern, request_obj):
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', '--weeks', type=float, default=0)
        parser.add_argument('-d', '--days', type=float, default=0)
        parser.add_argument('-H', '--hours', type=float, default=0)
        parser.add_argument('-m', '--minutes', type=float, default=0)
        parser.add_argument('-s', '--seconds', type=float, default=0)
        parser.add_argument('sentence', type=str)

        args = self._decompose_args(request_obj, parser, command_line, 'SCHEDULE')
        request_obj.text_answer = re.sub(regex_pattern, f'"{args.sentence}"', request_obj.text_answer)

        time_params = dict(weeks=args.weeks, days=args.days, hours=args.hours, minutes=args.minutes, seconds=args.seconds)
        run_date = datetime.now() + timedelta(**time_params)

        scheduled_request = RequestObject.copy(request_obj)
        scheduled_request.push = True
        scheduled_request.identifier = IdentityStore()[scheduled_request.socket_id]
        scheduled_request.text_answer = args.sentence
        TaskScheduler().add_task(lambda: self._dispatch(scheduled_request), run_date=run_date)

    def _on_draw(self, command_line, regex_pattern, request_obj):
        KeepAliveSet().add(request_obj.identifier)

        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch', type=int)
        parser.add_argument('-W', '--width', type=int)
        parser.add_argument('-H', '--height', type=int)
        parser.add_argument('-s', '--steps', dest='num_inference_steps', type=int)
        parser.add_argument('-g', '--guidance-scale', type=float)
        parser.add_argument('-m', '--mosaic', action='store_true')
        parser.add_argument('sentence', type=str)

        args = self._decompose_args(request_obj, parser, command_line, 'DRAW')
        request_obj.text_answer = re.sub(regex_pattern, f'"{args.sentence}"', request_obj.text_answer)
        self.img_intake.put(request_obj)

    def _on_query(self, command_line, regex_pattern, request_obj):
        KeepAliveSet().add(request_obj.identifier)

        parser = argparse.ArgumentParser()
        parser.add_argument('query', type=str)
        args = self._decompose_args(request_obj, parser, command_line, 'QUERY')
        request_obj.text_answer = re.sub(regex_pattern, f'"{args.query}"', request_obj.text_answer)

        if len(request_obj.indexes) > 0:
            try:
                request_obj.text_answer += '\n'
                for index in request_obj.indexes:
                    response = self._memoryManager.query_index(index, args.query, llm=request_obj.llm)
                    response = response._getvalue()
                    if response is not None:
                        sanitized_resp = str(response)
                        request_obj.text_answer += sanitized_resp
                        request_obj.text_answer += '\n'

                        self.chat_delegate.add_indexes_context(sanitized_resp, request_obj.preprompt, request_obj.llm)

                self._dispatch(request_obj)
            except Exception as e:
                err = RequestObject.copy(request_obj)
                err.text_answer = '<ERR>'
                err.silent = True
                err.priority = 0
                self._dispatch(err)

                err_request = RequestObject.copy(request_obj)
                err_request.text_answer = str(e)
                err_request.silent = True
                self._dispatch(err_request)

        term_req = KeepAliveSet().remove(request_obj.identifier)
        if term_req is not None:
            self._dispatch(term_req)

    def _on_search(self, command_line, regex_pattern, request_obj):
        KeepAliveSet().add(request_obj.identifier)

        parser = argparse.ArgumentParser()
        parser.add_argument('query', type=str)
        args = self._decompose_args(request_obj, parser, command_line, 'SEARCH')
        request_obj.text_answer = re.sub(regex_pattern, f'"{args.query}"', request_obj.text_answer)

        answers = ''
        try:
            responses = search(args.query, advanced=True, num_results=1)
            for res in responses:
                text = load_url(res.url)
                if text is not False:
                    answers += text

            if answers != '':
                google_request = RequestObject.copy(request_obj)
                google_request.silent = True
                google_request.text_request = answers
                google_request.user = 'Google Search Engine'
                self.chat_delegate._process_request(google_request)
            else:
                err = RequestObject.copy(request_obj)
                err.text_answer = '<ERR>'
                err.silent = True
                err.priority = 0
                self._dispatch(err)

                err_request = RequestObject.copy(request_obj)
                err_request.text_answer = 'Empty search results'
                err_request.silent = True
                self._dispatch(err_request)
        except Exception as e:
            err = RequestObject.copy(request_obj)
            err.text_answer = '<ERR>'
            err.silent = True
            err.priority = 0
            self._dispatch(err)

            err_request = RequestObject.copy(request_obj)
            err_request.text_answer = 'Unable to contact Google servers'
            err_request.silent = True
            self._dispatch(err_request)

        term_req = KeepAliveSet().remove(request_obj.identifier)
        if term_req is not None:
            self._dispatch(term_req)

    def _on_quiet(self, request_obj):
        termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)
        termination_request.priority = 0
        self._put(termination_request, request_obj.identifier)
        self.sio().emit('interrupt', Timer().now(), to=request_obj.socket_id)

    def _on_memory_wipe(self, request_obj):
        self.chat_delegate.clear_context(request_obj.preprompt)

        termination_request = RequestObject(request_obj.identifier, request_obj.user, termination=True)
        termination_request.priority = 0

        ack = RequestObject.copy(request_obj)
        ack.text_answer = '<MEMWIPE>'
        ack.silent = True

        self._put(ack, request_obj.identifier)
        self._put(termination_request, request_obj.identifier)

    def _decompose_args(self, request_obj, parser, command_line, command_name):
        cmd_token = self._commands[command_name].split(' "')[0]
        tokens = [t for t in shlex.split(command_line) if t != cmd_token]
        args = parser.parse_args(tokens)
        for k, v in vars(args).items():
            request_obj.command_args[k] = v
        return args

    def _argparsed_command(self, func, found_cmd, found_pattern, request_obj):
        try:
            func(found_cmd, found_pattern, request_obj)
        except (SystemExit, ValueError) as e:
            err = RequestObject.copy(request_obj)
            err.text_answer = '<ERR>'
            err.silent = True
            err.priority = 0
            self._dispatch(err)

            err_request = RequestObject.copy(request_obj)
            err_request.text_answer = 'Invalid arguments'
            err_request.silent = True
            self._dispatch(err_request)

# if __name__ == '__main__':
#     commands_file = ProjectPaths().resources_dir / 'default_sentences' / 'interpreted_commands.json'
#
#     with open(commands_file) as f:
#         commands = json.load(f)
#
#     pattern = commands['DRAW'][0]
#     pattern = fr'{pattern}'
#     # pattern = r'/draw "([^"]*)"'
#     text = 'Voilà ton truc de merde. /draw "Un poulet magique" Et maintenant arrêtes de me faire chier mec.'
#     resultat = re.search(pattern, text).group()
#     print(resultat)
#     texte_sans_match = re.sub(pattern, '', text)
#     print(texte_sans_match)