from hyperion.gui import UIAction
from hyperion.audio.io.sound_device_resource import SoundDeviceResource

import customtkinter
import tkinter as tk

from hyperion.utils.logger import ProjectLogger


class ParamsWindow(customtkinter.CTkToplevel):

    def __init__(self, x, y, out_queue, db_threshold, current_input_device, current_output_device, switch_camera):
        super().__init__()
        self.title('Parameters')
        self.geometry(f'325x180+{x}+{y}')
        self.resizable(False, False)
        self.wm_attributes('-topmost', True)

        self.out_queue = out_queue
        self.input_devices = SoundDeviceResource.list_devices('Input')
        self.output_devices = SoundDeviceResource.list_devices('Output')

        # self.grid_columnconfigure(1, weight=0)
        # self.grid_rowconfigure((0, 1, 2), weight=0)
        option_menu_width = 200
        label_left_pad = 10
        vertical_pad = 10

        self.input_label = customtkinter.CTkLabel(self, text='Input device:', width=100, anchor=tk.W)
        self.input_label.grid(row=0, column=0, padx=(label_left_pad, 5), pady=(vertical_pad, 0))
        self.input_device = customtkinter.CTkOptionMenu(self, values=list(self.input_devices.values()), width=option_menu_width, command=self.on_input_selected)
        self.input_device.set(current_input_device)
        self.input_device.grid(row=0, column=1, padx=(0, 10), pady=(vertical_pad, 0))

        self.output_label = customtkinter.CTkLabel(self, text='Output device:', width=100, anchor=tk.W)
        self.output_label.grid(row=1, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.output_device = customtkinter.CTkOptionMenu(self, values=list(self.output_devices.values()), width=option_menu_width, command=self.on_output_selected)
        self.output_device.set(current_output_device)
        self.output_device.grid(row=1, column=1, padx=(0, 10), pady=(5, 0))

        scaling = ['100%', '110%', '120%', '150%', '200%']
        self.scaling_label = customtkinter.CTkLabel(self, text='UI Scaling:', width=100, anchor=tk.W)
        self.scaling_label.grid(row=2, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self, values=scaling, width=option_menu_width, command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=2, column=1, padx=(0, 10), pady=(5, 0))

        self.db_slider_label = customtkinter.CTkLabel(self, text='dBs threshold:', width=100, anchor=tk.W)
        self.db_slider_label.grid(row=3, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.db_slider = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=100, command=self.on_threshold_change)
        self.db_slider.set(db_threshold)
        self.db_slider.grid(row=3, column=1)
        # self.db_slider.place(relx=10, rely=5, anchor=tk.W)

        self.video_label = customtkinter.CTkLabel(self, text='Enable video:', width=100, anchor=tk.W)
        self.video_label.grid(row=4, column=0, padx=(label_left_pad, 5), pady=(5, vertical_pad))
        self.video_switch = customtkinter.CTkSwitch(self, text='Camera', command=self.on_camera_switch)
        self.video_switch.grid(row=4, column=1, padx=(label_left_pad, 5), pady=(5, 0))
        if switch_camera:
            self.video_switch.select()
        else:
            self.video_switch.deselect()

    def on_camera_switch(self):
        self.out_queue.put((UIAction.CAMERA_SWITCH, self.video_switch.get()))

    def on_threshold_change(self, decibel):
        self.out_queue.put((UIAction.CHANGE_DB, decibel))

    def change_scaling_event(self, new_scaling):
        new_scaling_float = int(new_scaling.replace('%', '')) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def on_input_selected(self, selection):
        corresponding_idx = list(self.input_devices.keys())[list(self.input_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_INPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{self.input_devices} in {selection}')

    def on_output_selected(self, selection):
        corresponding_idx = list(self.output_devices.keys())[list(self.output_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_OUTPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{self.output_devices} in {selection}')
