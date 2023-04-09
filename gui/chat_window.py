from time import sleep
from PIL import Image

import os
import queue
import threading
import customtkinter
import tkinter as tk


customtkinter.set_appearance_mode('Dark')  # Modes: 'System' (standard), 'Dark', 'Light'
customtkinter.set_default_color_theme('blue')  # Themes: 'blue' (standard), 'green', 'dark-blue'


class ChatWindow(customtkinter.CTk):
    def __init__(self, bot_name, title='Chat window'):
        super().__init__()

        self._running = True
        self._in_message_queue = queue.Queue()
        self._out_message_queue = queue.Queue()
        self._previous_speaker = None
        self.bot_name = bot_name

        # configure window
        self.title(title)
        self.geometry(f'{450}x{250}')
        # self.config(bg='systemTransparent')
        self.attributes('-alpha', 0.95)
        # self.wm_attributes('-topmost', True)  # always on top
        # self.overrideredirect(True)  # hide title bar

        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(0, weight=1)

        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        image_dir = os.path.join(root_dir, 'resources', 'gui')
        self.trash_icon = customtkinter.CTkImage(Image.open(os.path.join(image_dir, 'trash.png')), size=(20, 20))
        self.gear_icon = customtkinter.CTkImage(Image.open(os.path.join(image_dir, 'settings.png')), size=(20, 20))

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, state=tk.DISABLED, border_color='#55595c', border_width=2)
        self.textbox.grid(row=0, column=1, columnspan=3, padx=(7, 7), pady=(10, 0), sticky='nsew')

        # color tags
        self.textbox.tag_config('bot', foreground='yellow')
        self.textbox.tag_config('author', foreground='cyan')

        self.entry = customtkinter.CTkEntry(self, placeholder_text='Send a message...')
        self.entry.grid(row=1, column=1, columnspan=1, padx=(7, 4), pady=(10, 10), sticky='nsew')
        self.entry.bind('<Return>', self.on_send)

        self.clear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.trash_icon, command=self.on_clear, hover_color='#313436')
        self.clear_button.grid(row=1, column=2, padx=(0, 0), pady=(10, 10), sticky='nsew')

        self.gear_button = customtkinter.CTkButton(self, fg_color='transparent', border_width=0, text='', width=20, image=self.gear_icon, command=self.on_gear, hover_color='#313436')
        self.gear_button.grid(row=1, column=3, padx=(0, 7), pady=(10, 10), sticky='nsew')

        self._handler = threading.Thread(target=self.message_handler, daemon=True)
        self._handler.start()

    def on_send(self, event):
        typed_message = self.entry.get()
        self.entry.delete('0', 'end')
        self._out_message_queue.put(typed_message)
        # self._insert_message('Unknown', typed_message)

    def on_clear(self):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.delete('0.0', tk.END)
        self.textbox.configure(state=tk.DISABLED)

    def on_gear(self):
        pass

    def _insert_message(self, author, message, with_delay=False):
        if author != self._previous_speaker:
            self._textbox_write(f'{author} : ', is_name=True)
        self._previous_speaker = author

        if with_delay:
            for char in message:
                self._textbox_write(char)
                sleep(0.03)
            self._textbox_write('\n')
        else:
            self._textbox_write(f'{message}\n')

    def _textbox_write(self, text, is_name=False):
        self.textbox.configure(state=tk.NORMAL)
        self.textbox.insert(tk.END, text)
        self.textbox.configure(state=tk.DISABLED)
        self.textbox.see(tk.END)  # AUTO SCROLL
        if is_name:
            lastline_index = self.textbox.index('end-1c linestart')
            line, col = lastline_index.split('.')
            tag = 'bot' if self.bot_name == text[:-3] else 'author'
            self.textbox.tag_add(tag, lastline_index, f'{line}.{len(text)-3}')

    def queue_message(self, author, message):
        self._in_message_queue.put((author, message))

    def drain_message(self):
        message = self._out_message_queue.get(timeout=0.1)
        self._out_message_queue.task_done()
        return message

    def message_handler(self):
        while self._running:
            try:
                author, text = self._in_message_queue.get(timeout=0.1)
                self._in_message_queue.task_done()
                self._insert_message(author, text, with_delay=True)
            except queue.Empty:
                continue


if __name__ == '__main__':
    app = ChatWindow('TOTO')
    app.mainloop()
