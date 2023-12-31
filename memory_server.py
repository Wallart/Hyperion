#!/usr/bin/env python
from hyperion.pipelines.memory import Memory
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager
from hyperion.utils.memory_utils import MANAGER_TOKEN

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion\'s memory server')
    parser.add_argument('--llama-host', type=str, default='localhost', help='Llama server host')
    parser.add_argument('--llama-port', type=int, default=8080, help='Llama server port')

    args = parser.parse_args()

    memory = Memory(llama_host=args.llama_host, llama_port=args.llama_port)
    manager = BaseManager(('', 5602), bytes(MANAGER_TOKEN, encoding='utf8'))
    manager.register('get_status', memory.get_status)
    manager.register('list_indexes', memory.list_indexes)
    manager.register('create_empty_index', memory.create_empty_index)
    manager.register('query_index', memory.query_index)
    manager.register('delete_index', memory.delete_index)
    manager.register('insert_into_index', memory.async_insert_into_index)
    manager.register('delete_from_index', memory.delete_from_index)
    manager.register('list_documents', memory.list_documents)
    server = manager.get_server()

    ProjectLogger().info('Memory server started.')
    server.serve_forever()
