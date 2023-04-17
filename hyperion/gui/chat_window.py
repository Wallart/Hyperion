from time import time, sleep
from PIL import Image, ImageTk
from hyperion.gui import UIAction
from pygments import lex, highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from hyperion.gui.params_window import ParamsWindow
from hyperion.gui.feedback_window import FeedbackWindow

import os
import json
import queue
import threading
import customtkinter
import tkinter as tk


customtkinter.set_appearance_mode('Dark')  # Modes: 'System' (standard), 'Dark', 'Light'
customtkinter.set_default_color_theme('blue')  # Themes: 'blue' (standard), 'green', 'dark-blue'


class ChatWindow(customtkinter.CTk):
    def __init__(self, bot_name, title='Chat window', savedir='~/.hyperion'):
        super().__init__()
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        image_dir = os.path.join(root_dir, 'resources', 'gui')

        self._gui_params = {}
        self._savefile = os.path.expanduser(os.path.join(savedir, 'gui_params.json'))
        if os.path.isfile(self._savefile):
            with open(self._savefile) as f:
                self._gui_params = json.load(f)

        self._running = True
        self._startup_time = time()

        self._in_message_queue = queue.Queue()
        self._out_message_queue = queue.Queue()
        self._previous_speaker = None
        self._previous_text = None
        self._code_block = False
        self._interrupt_stamp = 0
        self.bot_name = bot_name

        x = self._gui_params['x'] if 'x' in self._gui_params else 100
        y = self._gui_params['y'] if 'y' in self._gui_params else 300
        width = self._gui_params['width'] if 'width' in self._gui_params else 450
        height = self._gui_params['height'] if 'height' in self._gui_params else 250

        # configure window
        self.title(title)
        self.geometry(f'{width}x{height}+{x}+{y}')
        # self.config(bg='systemTransparent')
        self.attributes('-alpha', 0.95)
        # self.wm_attributes('-topmost', True)  # always on top
        # self.overrideredirect(True)  # hide title bar
        image = Image.open(os.path.join(image_dir, 'icon.png')).resize((512, 512))
        self.iconphoto(True, ImageTk.PhotoImage(image))
        self.bind('<Configure>', self.on_configure)
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.trash_icon = customtkinter.CTkImage(Image.open(os.path.join(image_dir, 'trash.png')), size=(20, 20))
        self.gear_icon = customtkinter.CTkImage(Image.open(os.path.join(image_dir, 'settings.png')), size=(20, 20))

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, state=tk.DISABLED, border_color='#55595c', border_width=2)
        self.textbox.grid(row=0, column=0, columnspan=4, padx=(7, 7), pady=(10, 0), sticky='nsew')

        # color tags
        self.textbox.tag_config('bot', foreground='#f2cb5a')
        self.textbox.tag_config('author', foreground='#2969d9')
        self.textbox.tag_config('pending', foreground='#989899')

        # code colors
        self.textbox.tag_config('Token.Keyword', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Constant', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Declaration', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Namespace', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Pseudo', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Reserved', foreground='#CC7A00')
        self.textbox.tag_config('Token.Keyword.Type', foreground='#CC7A00')
        self.textbox.tag_config('Token.Name.Builtin', foreground='#8888C6')
        self.textbox.tag_config('Token.Name.Class', foreground='#003D99')
        self.textbox.tag_config('Token.Name.Exception', foreground='#003D99')
        self.textbox.tag_config('Token.Name.Function', foreground='#003D99')
        self.textbox.tag_config('Token.Operator.Word', foreground='#CC7A00')
        self.textbox.tag_config('Token.Comment.Single', foreground='#B80000')
        self.textbox.tag_config('Token.Literal.String.Single', foreground='#248F24')
        self.textbox.tag_config('Token.Literal.String.Double', foreground='#248F24')

        self.name_entry = customtkinter.CTkEntry(self, placeholder_text='Username', width=100)
        self.name_entry.grid(row=1, column=0, columnspan=1, padx=(7, 0), pady=(10, 10), sticky='nsew')
        self.name_entry.bind('<FocusOut>', self.on_focus_out)
        if 'username' in self._gui_params:
            self.name_entry.insert(0, self._gui_params['username'])

        self.text_entry = customtkinter.CTkEntry(self, placeholder_text='Send a message...')
        self.text_entry.grid(row=1, column=1, columnspan=1, padx=(4, 4), pady=(10, 10), sticky='nsew')
        self.text_entry.bind('<Return>', self.on_send)

        self.clear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.trash_icon, command=self.on_clear, hover_color='#313436')
        self.clear_button.grid(row=1, column=2, padx=(0, 0), pady=(10, 10), sticky='nsew')

        self.gear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.gear_icon, command=self.on_gear, hover_color='#313436')
        self.gear_button.grid(row=1, column=3, padx=(0, 7), pady=(10, 10), sticky='nsew')

        self._message_thread = threading.Thread(target=self.message_handler)
        self._message_thread.start()

        self._video_thread = None

        self._params_window = None
        self._feedback_window = None
        self._frame_queue = None

        self.params_delegate = None

    def on_close(self):
        self._running = False
        self._out_message_queue.put((UIAction.QUIT,))
        self.destroy()

    def on_configure(self, event):
        if self._running and time() - self._startup_time >= 1:
            self._gui_params['x'] = event.x
            self._gui_params['y'] = event.y
            self._gui_params['width'] = event.width
            self._gui_params['height'] = event.height
            self._save_config()

    def on_focus_out(self, event):
        username = self.name_entry.get()
        if len(username) > 0:
            self._gui_params['username'] = username
            self._save_config()

    def on_send(self, event):
        username = self.name_entry.get()
        username = 'Unknown' if username == '' else username

        typed_message = self.text_entry.get()
        self.text_entry.delete('0', 'end')
        self._out_message_queue.put((UIAction.SEND_MESSAGE, username, typed_message))
        self._insert_message(time(), username, typed_message, pending=True)

    def on_clear(self):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.delete('0.0', tk.END)
        self.textbox.configure(state=tk.DISABLED)
        self._previous_speaker = None
        self._previous_text = None

    def on_gear(self):
        if self._params_window is None or not self._params_window.winfo_exists():
            db, input_dev, out_dev, camera_dev = self.params_delegate()
            x, y = self._gui_params['x'], self._gui_params['y']
            self._params_window = ParamsWindow(x, y, self._out_message_queue, db, input_dev, out_dev, camera_dev)
        else:
            self._params_window.focus()

    def _save_config(self):
        with open(self._savefile, 'w') as f:
            f.write(json.dumps(self._gui_params))

    def _insert_message(self, timestamp, author, message, with_delay=False, pending=False):
        if message == self._previous_text:
            # found = self.textbox.tag_ranges('pending')
            lastline_index = self.textbox.index('end-1c linestart')
            line, col = lastline_index.split('.')
            line = int(line) - 1
            self.textbox.tag_remove('pending', f'{line}.{col}', tk.END)
            return

        if author != self._previous_speaker:
            self._textbox_write(f'{author} : ', is_name=True, pending=pending)

        self._previous_speaker = author
        self._previous_text = message

        if with_delay:
            for char in message:
                if timestamp <= self._interrupt_stamp and self.bot_name == author:
                    break
                self._textbox_write(char)
                sleep(0.03)
        else:
            self._textbox_write(message, pending=pending)

        # self._colorize_code()
        self._textbox_write('\n')

    # def _colorize_code(self):
    #     lastline_index = self.textbox.index('end-1c linestart')
    #     lastline = self.textbox.get(lastline_index, tk.END)
    #     line, col = lastline_index.split('.')
    #
    #     if lastline.startswith('```') or lastline.startswith(f'{self._previous_speaker} : ```'):
    #         self._code_block = True
    #
    #     if self._code_block:
    #         for token, content in lex(lastline, PythonLexer()):
    #             start_idx = lastline.index(content)
    #             end_idx = start_idx + len(content)
    #             self.textbox.tag_add(str(token), f'{line}.{start_idx}', f'{line}.{end_idx}')
    #
    #     if lastline.endswith('```\n'):
    #         self._code_block = False

    def _textbox_write(self, text, is_name=False, pending=False):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.insert(tk.END, text)
        self.textbox.configure(state=tk.DISABLED)
        self.textbox.see(tk.END)  # AUTO SCROLL

        lastline_index = self.textbox.index('end-1c linestart')
        line, col = lastline_index.split('.')

        if is_name:
            tag = 'bot' if self.bot_name == text[:-3] else 'author'
            self.textbox.tag_add(tag, lastline_index, f'{line}.{len(text)-3}')

        if pending:
            # start_idx = 0#len(self._previous_speaker) + 3
            self.textbox.tag_add('pending', lastline_index, tk.END)

    def queue_message(self, timestamp, idx, requester, request, answer):
        self._in_message_queue.put((timestamp, idx, requester, request, answer))

    def drain_message(self):
        message = self._out_message_queue.get(timeout=0.1)
        self._out_message_queue.task_done()
        return message

    def mute(self, timestamp):
        # TODO not thread safe ?
        self._interrupt_stamp = timestamp

    def message_handler(self):
        while self._running:
            try:
                timestamp, idx, requester, request, answer = self._in_message_queue.get(timeout=0.1)
                self._in_message_queue.task_done()

                if timestamp <= self._interrupt_stamp:
                    continue

                if idx == 0:
                    self._insert_message(timestamp, requester, request)

                self._insert_message(timestamp, self.bot_name, answer, with_delay=True)
            except queue.Empty:
                continue

    def frame_handler(self):
        while True:
            try:
                frame = self._frame_queue.drain()
                if frame is None:
                    break

                if self._feedback_window is not None:
                    self._feedback_window.show(frame)
            except queue.Empty:
                continue

        # if running proper release
        if self._running:
            self._frame_queue = None
            self._feedback_window.destroy()
            self._feedback_window = None

    def set_camera_feedback(self, sink, width, height):
        if self._video_thread is not None:
            self._video_thread.join()

        if self._feedback_window is None or not self._feedback_window.winfo_exists():
            self._feedback_window = FeedbackWindow(width, height)
        else:
            self._feedback_window.focus()

        if self._frame_queue is None:
            self._frame_queue = sink

        self._video_thread = threading.Thread(target=self.frame_handler)
        self._video_thread.start()


if __name__ == '__main__':
    app = ChatWindow('TOTO')
    app.mainloop()
