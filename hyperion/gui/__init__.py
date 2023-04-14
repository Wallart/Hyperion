from enum import Enum


class UIAction(Enum):
    QUIT = 0
    SEND_MESSAGE = 1
    CHANGE_INPUT_DEVICE = 2
    CHANGE_OUTPUT_DEVICE = 3
    CHANGE_DB = 4
    CAMERA_SWITCH = 5
