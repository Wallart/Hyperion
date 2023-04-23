from time import time, sleep
from PIL import Image, ImageTk
from hyperion.gui import UIAction
from hyperion.utils.timer import Timer
from hyperion.utils import ProjectPaths
from hyperion.gui.gui_params import GUIParams
from hyperion.analysis import sanitize_username
from hyperion.gui.params_window import ParamsWindow
from hyperion.gui.code_formatter import CodeFormatter
from hyperion.gui.feedback_window import FeedbackWindow

import queue
import threading
import customtkinter
import tkinter as tk


customtkinter.set_appearance_mode('Dark')  # Modes: 'System' (standard), 'Dark', 'Light'
customtkinter.set_default_color_theme('blue')  # Themes: 'blue' (standard), 'green', 'dark-blue'


class ChatWindow(customtkinter.CTk):
    def __init__(self, bot_name, prompts, current_prompt, llms, current_llm, title='Chat window'):
        super().__init__()

        image_dir = ProjectPaths().resources_dir / 'gui'

        self._running = True
        self._startup_time = time()

        self._in_message_queue = queue.Queue()
        self._out_message_queue = queue.Queue()
        self._previous_speaker = None
        self._previous_text = None

        self._interrupt_stamp = 0
        self.bot_name = bot_name

        x = GUIParams()['x'] if 'x' in GUIParams() else 100
        y = GUIParams()['y'] if 'y' in GUIParams() else 300
        width = GUIParams()['width'] if 'width' in GUIParams() else 450
        height = GUIParams()['height'] if 'height' in GUIParams() else 250

        # configure window
        self.title(title)
        self.geometry(f'{width}x{height}+{x}+{y}')
        # self.config(bg='systemTransparent')
        self.attributes('-alpha', 0.95)
        # self.wm_attributes('-topmost', True)  # always on top
        # self.overrideredirect(True)  # hide title bar
        image = Image.open(image_dir / 'icon.png').resize((512, 512))
        self.iconphoto(True, ImageTk.PhotoImage(image))
        self.bind('<Configure>', self.on_configure)
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.trash_icon = customtkinter.CTkImage(Image.open(image_dir / 'trash.png'), size=(20, 20))
        self.gear_icon = customtkinter.CTkImage(Image.open(image_dir /'settings.png'), size=(20, 20))

        # 🟢🟠🔴
        self.status_label = customtkinter.CTkLabel(self, text='Status: 🔴', width=100, anchor=tk.W)
        self.status_label.grid(row=0, column=0, padx=(7, 0), pady=(10, 0))

        opts = dict(width=100, dynamic_resizing=False)
        # llm selection
        self.llm_optionemenu = customtkinter.CTkOptionMenu(self, values=llms, command=self.on_llm_change, **opts)
        self.llm_optionemenu.grid(row=0, column=1, columnspan=1, padx=(7, 0), pady=(10, 0), sticky=tk.E)
        self.llm_optionemenu.set(current_llm)

        # preprompt selection
        self.preprompt_optionemenu = customtkinter.CTkOptionMenu(self, values=prompts, command=self.on_prompt_change, **opts)
        self.preprompt_optionemenu.grid(row=0, column=2, columnspan=3, padx=(7, 7), pady=(10, 0))
        self.preprompt_optionemenu.set(current_prompt)

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, state=tk.DISABLED, border_color='#55595c', border_width=2)
        self.textbox.grid(row=1, column=0, columnspan=5, padx=(7, 7), pady=(10, 0), sticky=tk.NSEW)

        # color tags
        self.textbox.tag_config('bot', foreground='#f2cb5a')
        self.textbox.tag_config('author', foreground='#2969d9')
        self.textbox.tag_config('pending', foreground='#989899')

        self._formatter = CodeFormatter(self.textbox, self.bot_name)

        self.name_entry = customtkinter.CTkEntry(self, placeholder_text='Username', width=100)
        self.name_entry.grid(row=2, column=0, columnspan=1, padx=(7, 0), pady=(10, 10), sticky=tk.NSEW)
        self.name_entry.bind('<FocusOut>', self.on_focus_out)
        if 'username' in GUIParams():
            self.name_entry.insert(0, GUIParams()['username'])

        self.text_entry = customtkinter.CTkEntry(self, placeholder_text='Send a message...')
        self.text_entry.grid(row=2, column=1, columnspan=2, padx=(4, 4), pady=(10, 10), sticky=tk.NSEW)
        self.text_entry.bind('<Return>', self.on_send)

        self.clear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.trash_icon, command=self.on_clear, hover_color='#313436')
        self.clear_button.grid(row=2, column=3, padx=(0, 0), pady=(10, 10), sticky=tk.NSEW)

        self.gear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.gear_icon, command=self.on_gear, hover_color='#313436')
        self.gear_button.grid(row=2, column=4, padx=(0, 7), pady=(10, 10), sticky=tk.NSEW)

        self._message_thread = threading.Thread(target=self.message_handler)
        self._message_thread.start()

        self._video_thread = None

        self._params_window = None
        self._feedback_window = None
        self._frame_queue = None

        self.params_delegate = None

        if 'scaling' in GUIParams():
            ParamsWindow.rescale_gui(GUIParams()['scaling'])

    def update_status(self, state):
        if state == 'online':
            state = '🟢'
        elif state == 'offline':
            state = '🔴'
        elif state == 'busy':
            state = '🟠'

        self.status_label.configure(text=f'Status: {state}')

    def on_close(self):
        self._running = False
        self._out_message_queue.put((UIAction.QUIT,))
        self.destroy()

    def on_llm_change(self, selection):
        self._out_message_queue.put((UIAction.CHANGE_LLM, selection))

    def on_prompt_change(self, selection):
        self._out_message_queue.put((UIAction.CHANGE_PROMPT, selection))

    def on_configure(self, event):
        if self._running and time() - self._startup_time >= 1:
            GUIParams()['x'] = event.x
            GUIParams()['y'] = event.y
            GUIParams()['width'] = event.width
            GUIParams()['height'] = event.height
            GUIParams().save()

    def on_focus_out(self, event):
        username = sanitize_username(self.name_entry.get())
        self.name_entry.delete(0, tk.END)
        if username is not None and len(username) > 0:
            self.name_entry.insert(0, username)
            GUIParams()['username'] = username
        else:
            del GUIParams()['username']

        GUIParams().save()

    def on_send(self, event):
        username = self.name_entry.get()
        username = 'Unknown' if username == '' else username

        typed_message = self.text_entry.get()
        self.text_entry.delete(0, tk.END)
        self._out_message_queue.put((UIAction.SEND_MESSAGE, username, typed_message))
        self._interrupt_stamp = Timer().now()  # will force flush currently writing messages
        self.queue_message(None, None, username, typed_message, None)

    def on_clear(self):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.delete(0.0, tk.END)
        self.textbox.configure(state=tk.DISABLED)
        self._previous_speaker = None
        self._previous_text = None

    def on_gear(self):
        if self._params_window is None or not self._params_window.winfo_exists():
            db, input_dev, out_dev, camera_dev = self.params_delegate()
            x, y = GUIParams()['x'], GUIParams()['y']
            self._params_window = ParamsWindow(x, y, self._out_message_queue, db, input_dev, out_dev, camera_dev)
        else:
            self._params_window.focus()

    def _remove_pending(self, message):
        if message != self._previous_text:
            return

        # found = self.textbox.tag_ranges('pending')
        lastline_index = self.textbox.index('end-1c linestart')
        line, col = lastline_index.split('.')
        line = int(line) - 1
        self.textbox.tag_remove('pending', f'{line}.{col}', tk.END)

    def _insert_message(self, timestamp, author, message, with_delay=False, pending=False):
        if author != self._previous_speaker:
            self._textbox_write(f'{author} : ', is_name=True, pending=pending)

        self._previous_speaker = author
        self._previous_text = message

        if with_delay:
            for i, char in enumerate(message):
                # if timestamp <= self._interrupt_stamp and self.bot_name == author:
                if not Timer().gt(timestamp, self._interrupt_stamp) and self.bot_name == author:
                    self._textbox_write(message[i:])  # flush current message
                    break
                self._textbox_write(char)
                sleep(0.01)
        else:
            self._textbox_write(message, pending=pending)

        self._formatter.colorize(self._previous_speaker)
        self._textbox_write('\n')

    def _textbox_write(self, text, is_name=False, pending=False):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.insert(tk.END, text)
        self.textbox.configure(state=tk.DISABLED)
        _, end_val = self.textbox._y_scrollbar.get()
        if end_val == 1.0:
            # AUTO SCROLL only if scrollbar is at the bottom
            self.textbox.see(tk.END)

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

                if idx is None and answer is None:
                    self._insert_message(timestamp, requester, request, pending=True)
                else:
                    # if timestamp <= self._interrupt_stamp:
                    if not Timer().gt(timestamp, self._interrupt_stamp):
                        continue

                    if idx == 0:
                        self._remove_pending(request)
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
    app = ChatWindow('TOTO', ['base', 'gamemaster'], 'base', ['gpt-3.5', 'gpt-4'], 'gpt-3.5')
    app.params_delegate = lambda: (130, 'input_dev', 'out_dev', 'camera_dev')
    app.mainloop()
