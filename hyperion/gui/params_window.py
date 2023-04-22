from hyperion.gui import UIAction
from hyperion.video.io import VideoDevices
from hyperion.gui.gui_params import GUIParams
from hyperion.utils.logger import ProjectLogger
from hyperion.audio.io.sound_device_resource import SoundDeviceResource

import customtkinter
import tkinter as tk


class ParamsWindow(customtkinter.CTkToplevel):

    def __init__(self, x, y, out_queue, db_threshold, cur_input_dev, cur_output_dev, cur_camera_dev):
        super().__init__()
        self.title('Parameters')
        self.geometry(f'325x180+{x}+{y}')
        self.resizable(False, False)
        self.wm_attributes('-topmost', True)

        self.out_queue = out_queue
        self.cameras = ['Disabled'] + VideoDevices().list_devices()
        self.input_devices = SoundDeviceResource.list_devices('Input')
        self.output_devices = SoundDeviceResource.list_devices('Output')

        self.grid_columnconfigure(1, weight=1)
        # self.grid_rowconfigure(4, weight=1)
        option_menu_width = 200
        label_left_pad = 10
        vertical_pad = 10

        common_opt_menu = dict(width=option_menu_width, dynamic_resizing=False)

        self.input_label = customtkinter.CTkLabel(self, text='Input device:', width=100, anchor=tk.W)
        self.input_label.grid(row=0, column=0, padx=(label_left_pad, 5), pady=(vertical_pad, 0))
        self.input_device = customtkinter.CTkOptionMenu(self, values=list(self.input_devices.values()), command=self.on_input_selected, **common_opt_menu)
        self.input_device.set(cur_input_dev)
        self.input_device.grid(row=0, column=1, padx=(0, 10), pady=(vertical_pad, 0))

        self.output_label = customtkinter.CTkLabel(self, text='Output device:', width=100, anchor=tk.W)
        self.output_label.grid(row=1, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.output_device = customtkinter.CTkOptionMenu(self, values=list(self.output_devices.values()), command=self.on_output_selected, **common_opt_menu)
        self.output_device.set(cur_output_dev)
        self.output_device.grid(row=1, column=1, padx=(0, 10), pady=(5, 0))

        self.video_label = customtkinter.CTkLabel(self, text='Video input:', width=100, anchor=tk.W)
        self.video_label.grid(row=2, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.video_device = customtkinter.CTkOptionMenu(self, values=self.cameras, command=self.on_camera_change, **common_opt_menu)
        self.video_device.set(self.cameras[0] if cur_camera_dev == -1 else cur_camera_dev)
        self.video_device.grid(row=2, column=1, padx=(0, 10), pady=(5, 0))

        scaling = ['100%', '110%', '120%', '150%', '200%']
        self.scaling_label = customtkinter.CTkLabel(self, text='UI Scaling:', width=100, anchor=tk.W)
        self.scaling_label.grid(row=3, column=0, padx=(label_left_pad, 5), pady=(5, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self, values=scaling, command=self.change_scaling_event, **common_opt_menu)
        self.scaling_optionemenu.grid(row=3, column=1, padx=(0, 10), pady=(5, 0))
        if 'scaling' in GUIParams():
            self.scaling_optionemenu.set(GUIParams()['scaling'])

        self.db_slider_label = customtkinter.CTkLabel(self, text='dBs threshold:', width=100, anchor=tk.W)
        self.db_slider_label.grid(row=4, column=0, padx=(label_left_pad, 5), pady=(5, vertical_pad))
        self.db_slider = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=100, command=self.on_threshold_change)
        self.db_slider.grid(row=4, column=1, padx=(0, 10), pady=(0, vertical_pad))
        self.db_slider.set(db_threshold)

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
        self.out_queue.put((UIAction.CHANGE_DB, decibel))

    def change_scaling_event(self, new_scaling):
        ParamsWindow.rescale_gui(new_scaling)
        GUIParams()['scaling'] = new_scaling
        GUIParams().save()

    def on_input_selected(self, selection):
        corresponding_idx = list(self.input_devices.keys())[list(self.input_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_INPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{self.input_devices} in {selection}')

    def on_output_selected(self, selection):
        corresponding_idx = list(self.output_devices.keys())[list(self.output_devices.values()).index(selection)]
        self.out_queue.put((UIAction.CHANGE_OUTPUT_DEVICE, corresponding_idx))
        ProjectLogger().info(f'{self.output_devices} in {selection}')
