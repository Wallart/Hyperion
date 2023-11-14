from enum import Enum
from time import time
from hyperion.utils.paths import ProjectPaths
from hyperion.utils.logger import ProjectLogger
from hyperion.utils.request import RequestObject
from hyperion.utils.threading import Consumer, Producer

import re
import json
import shlex
import queue
import argparse


class ACTIONS(Enum):
    DRAW = 0


class InterpretedCommandDetector(Consumer, Producer):
    def __init__(self):
        super().__init__()
        self.img_delegate = None
        self.img_intake = None
        self._commands_file = ProjectPaths().resources_dir / 'default_sentences' / 'interpreted_commands.json'
        self._cmd_buffer = ''

        with open(self._commands_file) as f:
            self._commands = json.load(f)

    def set_img_delegate(self, img_delegate):
        self.img_delegate = img_delegate
        self.img_intake = img_delegate.get_intake()

    def run(self):
        while self.running:
            try:
                request_obj = self._consume()
                t0 = time()

                not_term = request_obj.termination is False
                not_pending = request_obj.identifier not in self.img_delegate.keep_alive
                # don't forward termination requests if there is work still pending
                if request_obj.text_answer is None:
                    if not_term or not_pending:
                        self._dispatch(request_obj)
                    continue

                text_answer = self._cmd_buffer + request_obj.text_answer

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
                    if res is not None and comma_count < 2:
                        self._cmd_buffer = text_answer
                        break

                if action is None and self._cmd_buffer == '':
                    self._dispatch(request_obj)
                elif action is not None:
                    ProjectLogger().info(f'Command found in "{found_cmd}"')

                    ack = RequestObject.copy(request_obj)
                    ack.text_answer = '<CMD>'
                    ack.silent = True
                    ack.priority = 0
                    self._dispatch(ack)

                    if action == ACTIONS.DRAW.value:
                        self._on_draw(found_cmd, found_pattern, request_obj)

                ProjectLogger().info(f'{self.__class__.__name__} {time() - t0:.3f} COMMAND exec. time')
            except queue.Empty:
                continue

        ProjectLogger().info('Interpreted Command Detector stopped.')

    def _on_draw(self, command_line, regex_pattern, request_obj):
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch', type=int)
        parser.add_argument('-W', '--width', type=int)
        parser.add_argument('-H', '--height', type=int)
        parser.add_argument('-s', '--steps', dest='num_inference_steps', type=int)
        parser.add_argument('-g', '--guidance-scale', type=float)
        parser.add_argument('-m', '--mosaic', action='store_true')
        parser.add_argument('sentence', type=str)

        try:
            tokens = [t for t in shlex.split(command_line) if t != '/draw']
            args = parser.parse_args(tokens)
            for k, v in vars(args).items():
                request_obj.command_args[k] = v

            request_obj.text_answer = re.sub(regex_pattern, f'"{args.sentence}"', request_obj.text_answer)
            self.img_intake.put(request_obj)
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