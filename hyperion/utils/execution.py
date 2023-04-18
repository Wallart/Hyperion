from functools import partial
from daemonocle import Daemon
from . import ProjectPaths
from .logger import ProjectLogger

import os
import sys
import signal
import traceback


def handle_errors(fn):
    def wrapper(app_name, opts):
        try:
            res = fn(opts)
            return res
        except Exception as e:
            ProjectLogger().error(f'Fatal error occurred : {e}')
            # name = os.path.basename(sys.argv[0])
            traceback_file = ProjectPaths().log_dir / f'{app_name.lower()}-error.log'
            ProjectLogger().error(f'Traceback saved at {traceback_file}')
            with open(traceback_file, 'w') as f:
                f.write(traceback.format_exc())
    return wrapper


def open_pid(pid_file):
    with open(pid_file, 'r') as f:
        pid = [l.strip() for l in f.readlines()]
    return pid


def kill(pid_file):
    pid_value = open_pid(pid_file)
    if len(pid_value) > 0:
        try:
            pid_value = int(pid_value[0])
            os.kill(pid_value, signal.SIGKILL)
        except Exception as e:
            ProjectLogger().warning(f'Invalid pid number in {pid_file}')
        pid_file.unlink()


def startup(app_name, parser, fn):
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        exit(1)

    opts = parser.parse_args()
    _ = ProjectLogger(opts, app_name)

    pid_file = ProjectPaths().pid_dir / f'{app_name.lower()}.pid'
    if hasattr(opts, 'action'):
        if opts.action == 'stop' and pid_file.exists():
            kill(pid_file)
        elif opts.action in ['start', 'restart']:
            if pid_file.exists():
                if opts.action == 'restart':
                    ProjectLogger().warning('Already running. Restarting app...')
                    kill(pid_file)
                else:
                    ProjectLogger().error('Already running.')
                    exit(1)

            if opts.daemon:
                daemon = Daemon(worker=partial(fn, app_name, opts), pid_file=pid_file)
                daemon.do_action('start')
            else:
                return fn(app_name, opts)
    else:
        # TODO Fix this ugly piece of duplicated code
        if opts.daemon:
            daemon = Daemon(worker=partial(fn, app_name, opts), pid_file=pid_file)
            daemon.do_action('start')
        else:
            return fn(app_name, opts)
