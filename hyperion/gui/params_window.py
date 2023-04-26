from tktooltip import ToolTip
from hyperion.gui import UIAction
from hyperion.video.io import VideoDevices
from hyperion.gui.gui_params import GUIParams
from hyperion.gui.audio_level import AudioLevel
from hyperion.utils.logger import ProjectLogger
from hyperion.audio.io.sound_device_resource import SoundDeviceResource

import customtkinter
import tkinter as tk


class ParamsWindow(customtkinter.CTkToplevel):

    def __init__(self, x, y, out_queue, db_threshold, cur_input_dev, cur_output_dev, cur_camera_dev):
        super().__init__()
        win_width = 330
        win_height = 220

        self.title('Parameters')
        self.geometry(f'{win_width}x{win_height}+{x}+{y}')
        self.resizable(False, False)

        self.out_queue = out_queue
        self.cameras = ['Disabled'] + VideoDevices().list_devices()
        self.input_devices = SoundDeviceResource.list_devices('Input')
        self.output_devices = SoundDeviceResource.list_devices('Output')
        self.db_threshold = db_threshold
        if 'db' in GUIParams():
            self.db_threshold = GUIParams()['db']

        self.grid_columnconfigure(1, weight=1)
        # self.grid_rowconfigure(4, weight=1)
        label_width = 100
        option_width = 200

        outer_horizontal_pad = 10
        inner_horizontal_pad = 2.5
        outer_vertical_pad = 10
        inner_vertical_pad = 2.5

        col_0_padx = (outer_horizontal_pad, inner_horizontal_pad)
        col_1_padx = (inner_horizontal_pad, outer_horizontal_pad)

        row_0_pady = (outer_vertical_pad, inner_vertical_pad)
        row_x_pady = (inner_vertical_pad, inner_vertical_pad)
        row_n_pad_y = (inner_vertical_pad, outer_vertical_pad)

        common_opt_menu = dict(width=option_width, dynamic_resizing=False)

        self.input_label = customtkinter.CTkLabel(self, text='Input device:', width=label_width, anchor=tk.W)
        self.input_label.grid(row=0, column=0, padx=col_0_padx, pady=row_0_pady)
        self.input_device = customtkinter.CTkOptionMenu(self, values=list(self.input_devices.values()), command=self.on_input_selected, **common_opt_menu)
        self.input_device.set(cur_input_dev)
        self.input_device.grid(row=0, column=1, padx=col_1_padx, pady=row_0_pady)

        self.output_label = customtkinter.CTkLabel(self, text='Output device:', width=label_width, anchor=tk.W)
        self.output_label.grid(row=1, column=0, padx=col_0_padx, pady=row_x_pady)
        self.output_device = customtkinter.CTkOptionMenu(self, values=list(self.output_devices.values()), command=self.on_output_selected, **common_opt_menu)
        self.output_device.set(cur_output_dev)
        self.output_device.grid(row=1, column=1, padx=col_1_padx, pady=row_x_pady)

        self.video_label = customtkinter.CTkLabel(self, text='Video input:', width=label_width, anchor=tk.W)
        self.video_label.grid(row=2, column=0, padx=col_0_padx, pady=row_x_pady)
        self.video_device = customtkinter.CTkOptionMenu(self, values=self.cameras, command=self.on_camera_change, **common_opt_menu)
        self.video_device.set(self.cameras[0] if cur_camera_dev == -1 else cur_camera_dev)
        self.video_device.grid(row=2, column=1, padx=col_1_padx, pady=row_x_pady)

        scaling = ['100%', '110%', '120%', '150%', '200%']
        self.scaling_label = customtkinter.CTkLabel(self, text='UI Scaling:', width=label_width, anchor=tk.W)
        self.scaling_label.grid(row=3, column=0, padx=col_0_padx, pady=row_x_pady)
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self, values=scaling, command=self.change_scaling_event, **common_opt_menu)
        self.scaling_optionemenu.grid(row=3, column=1, padx=col_1_padx, pady=row_x_pady)
        if 'scaling' in GUIParams():
            self.scaling_optionemenu.set(GUIParams()['scaling'])

        self.db_slider_label = customtkinter.CTkLabel(self, text='Input volume:', width=label_width, anchor=tk.W)
        self.db_slider_label.grid(row=4, column=0, padx=col_0_padx, pady=row_x_pady)
        self.db_slider = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=100, width=option_width, command=self.on_threshold_change)
        self.db_slider.grid(row=4, column=1, padx=col_1_padx, pady=row_x_pady)
        self.db_slider.set(self.db_threshold)
        self.db_slider_tooltip = ToolTip(self.db_slider, msg=self.slider_tooltip_msg, delay=1.0)

        # levels_width = win_width - label_width - 2 * outer_horizontal_pad - 5
        self.levels_label = customtkinter.CTkLabel(self, text='Input level:', width=label_width, anchor=tk.W)
        self.levels_label.grid(row=5, column=0, padx=col_0_padx, pady=row_n_pad_y)
        self.levels = AudioLevel(self, width=option_width, height=20)
        self.levels.grid(row=5, column=1, padx=col_1_padx, pady=row_n_pad_y)

    @staticmethod
    def rescale_gui(new_scaling: str):
        new_scaling_float = int(new_scaling.replace('%', '')) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
        customtkinter.set_window_scaling(new_scaling_float)

    def on_camera_change(self, selection):
        if selection == self.cameras[0]:
            self.out_queue.put((UIAction.DISABLED_CAMERA_DEVICE,))
        else:
            self.out_queue.put((UIAction.CHANGE_CAMERA_DEVICE, selection))

    def on_threshold_change(self, decibel):
        self.db_threshold = decibel
        self.out_queue.put((UIAction.CHANGE_DB, decibel))
        GUIParams()['db'] = decibel
        GUIParams().save()

    def change_scaling_event(self, new_scaling):
        ParamsWindow.rescale_gui(new_scaling)
        GUIParams()['scaling'] = new_scaling
        GUIParams().save()

    def on_input_selected(self, selection):
        corresponding_idx = list(self.input_devices.keys())[list(self.input_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_INPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{selection} in {self.input_devices}')

    def on_output_selected(self, selection):
        corresponding_idx = list(self.output_devices.keys())[list(self.output_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_OUTPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{selection} in {self.output_devices}')

    def slider_tooltip_msg(self):
        return f'{self.db_threshold} dBs'
