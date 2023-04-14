from PIL import Image, ImageTk

import customtkinter
import tkinter as tk


class FeedbackWindow(customtkinter.CTkToplevel):
    def __init__(self, width, height):
        super().__init__()
        self.title('Video feedback')
        self.geometry(f'{width}x{height}')
        self.resizable(False, False)
        # self.overrideredirect(True)  # hide title bar

        self.width = width
        self.height = height

        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=1, padx=0, pady=0, sticky='nsew')

        # To avoid garbage collection
        self._current_img = None

    def show(self, img_array):
        # img = Image.open('/Users/wallart/Desktop/sing.jpeg').resize((self.width, self.height))
        # self._current_img = ImageTk.PhotoImage(img)
        self._current_img = ImageTk.PhotoImage(image=Image.fromarray(img_array))
        self.canvas.create_image(0, 0, state=tk.NORMAL, image=self._current_img, anchor=tk.NW)
