from hyperion.utils.logger import ProjectLogger

import torch


def get_ctx(args):
    devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
    if torch.cuda.is_available():
        if len(devices_id) == 0:
            devices_id = list(range(torch.cuda.device_count()))

        ctx = [torch.device(f'cuda:{i}') for i in devices_id if i >= 0]
        ctx = ctx if len(ctx) > 0 else [torch.device('cpu')]
    else:
        ProjectLogger().warning('Cannot access GPU.')
        ctx = [torch.device('cpu')]

    ProjectLogger().info('Used context: {}'.format(', '.join([str(x) for x in ctx])))
    return ctx


def load_file(path, strip=True):
    with open(path) as f:
        content = f.readlines()
    if strip:
        return [l.strip() for l in content]

    return content
