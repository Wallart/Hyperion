from pygments import lex
from hyperion.utils.logger import ProjectLogger

import re
import importlib
import tkinter as tk


class CodeFormatter:

    def __init__(self, textbox, bot_name):
        self.code_tag = '```'
        self.bot_name = bot_name
        self.textbox = textbox
        self.lexers = importlib.import_module('pygments.lexers')

        # code colors
        self.textbox.tag_config('Token.Keyword', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Constant', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Declaration', foreground='#FFC66D')
        self.textbox.tag_config('Token.Keyword.Namespace', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Pseudo', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Reserved', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Type', foreground='#CC7A00')
        self.textbox.tag_config('Token.Name.Builtin', foreground='#8888C6')
        self.textbox.tag_config('Token.Name.Builtin.Pseudo', foreground='#94558D')  # self
        self.textbox.tag_config('Token.Name.Class', foreground='#FFFFFF')
        self.textbox.tag_config('Token.Name.Exception', foreground='#003D99')
        self.textbox.tag_config('Token.Name.Function', foreground='#FFC66D')
        self.textbox.tag_config('Token.Name.Decorator', foreground='#BBB529')
        self.textbox.tag_config('Token.Name.Variable.Magic', foreground='#B200B2')
        self.textbox.tag_config('Token.Name.Function.Magic', foreground='#B200B2')
        self.textbox.tag_config('Token.Operator.Word', foreground='#CC7A00')
        self.textbox.tag_config('Token.Operator', foreground='#CC7A00')
        self.textbox.tag_config('Token.Comment.Single', foreground='#808080')
        self.textbox.tag_config('Token.Comment.Preproc', foreground='#808080')
        self.textbox.tag_config('Token.Comment.PreprocFile', foreground='#B80000')
        # strings
        self.textbox.tag_config('Token.Literal.String.Interpol', foreground='#CC7A00')  # f'{}'
        self.textbox.tag_config('Token.Literal.String.Escape', foreground='#CC7A00')  # \n
        self.textbox.tag_config('Token.Literal.String.Affix', foreground='#629755')  # f''
        self.textbox.tag_config('Token.Literal.String.Doc', foreground='#629755')  # multiline comments
        self.textbox.tag_config('Token.Literal.String', foreground='#629755')
        self.textbox.tag_config('Token.Literal.String.Single', foreground='#629755')
        self.textbox.tag_config('Token.Literal.String.Double', foreground='#629755')
        # numbers
        self.textbox.tag_config('Token.Literal.Number.Integer', foreground='#6897BB')
        self.textbox.tag_config('Token.Literal.Number.Float', foreground='#6897BB')

    def colorize(self, prev_speaker):
        try:
            blocks = self.textbox.tag_ranges('bot') if prev_speaker == self.bot_name else self.textbox.tag_ranges('author')
            start_block = blocks[-2]

            lines = self.textbox.get(start_block, tk.END).split('\n')
            start_line, start_col = str(start_block).split('.')

            code_block = False
            language = None
            for i, current_line in enumerate(lines):
                line_idx = int(start_line) + i
                len_author = len(f'{prev_speaker} : ') if i == 0 else 0

                start_idx = len_author
                end_idx = len(current_line)
                occurences = [i.start() for i in re.finditer(self.code_tag, current_line)]

                if len(occurences) == 0 and not code_block:
                    continue
                elif len(occurences) == 1:
                    if code_block:
                        end_idx = occurences[0]
                    else:
                        start_idx = occurences[0] + 3
                        language = current_line[start_idx:].capitalize()
                    code_block = not code_block
                elif len(occurences) == 2:
                    start_idx = occurences[0] + 3
                    end_idx = occurences[1]

                self.parse_line(line_idx, current_line, start_idx, end_idx, language)
        except Exception as e:
            ProjectLogger().error(e)

    def parse_line(self, line_idx, current_line, parse_start_idx, parse_end_idx, language=None):
        cursor = 0
        work_area = current_line[parse_start_idx:parse_end_idx]

        if language is not None and hasattr(self.lexers, f'{language}Lexer'):
            clazz = getattr(self.lexers, f'{language}Lexer')
        else:
            clazz = getattr(self.lexers, f'PythonLexer')

        for token, content in lex(work_area, clazz()):
            if content == '\n':
                continue
            token_start_idx = [i.start() for i in re.finditer(re.escape(content), current_line)]
            token_start_idx = list(filter(lambda e: e >= cursor, token_start_idx))[0]
            cursor = token_start_idx + 1
            token_end_idx = token_start_idx + len(content)
            # print(current_line[token_start_idx:token_end_idx])
            self.textbox.tag_add(str(token), f'{line_idx}.{token_start_idx}', f'{line_idx}.{token_end_idx}')
