#!/usr/bin/env python

from utils.utils import get_ctx
from utils.logger import ProjectLogger
from pipelines.listener import Listener
from utils.execution import startup, handle_errors

import os
import argparse
import socketio


APP_NAME = os.path.basename(__file__).split('.')[0]
sio = socketio.Client()


@sio.on('interrupt')
def on_interrupt():
    listener.interrupt()


@sio.event
def connect():
    sid = sio.get_sid()
    ProjectLogger().info(f'Connection opened with SID : {sid}')
    listener.sid = sid


@sio.event
def connect_error(data):
    ProjectLogger().warning('Connection failed !')


@sio.event
def disconnect():
    ProjectLogger().warning('Disconnected.')


@handle_errors
def main(args):
    global listener
    ctx = get_ctx(args)
    listener = Listener(ctx, args)
    ProjectLogger().info(f'Opening connection to {args.target_url}')
    sio.connect(f'ws://{args.target_url}')
    listener.start(sio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s local ear')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon')
    parser.add_argument('--debug', action='store_true', help='Enables debugging.')
    parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')

    parser.add_argument('--in-idx', type=int, default=-1, help='Audio input identifier')
    parser.add_argument('--rms', type=int, default=1000, help='Sound detection threshold')
    parser.add_argument('--out-idx', type=int, default=-1, help='Audio output identifier')
    parser.add_argument('--target-url', type=str, default='localhost:9999', help='Brain target URL')
    parser.add_argument('--dummy-file', type=str, help='Play file instead of Brain\'s responses')
    parser.add_argument('--recog', action='store_true', help='Start bot with local user recognition. Slower than server recognition if there is no GPU.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')

    startup(APP_NAME.lower(), parser, main)
