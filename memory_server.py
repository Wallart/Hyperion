from hyperion.pipelines.memory import Memory
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager
from hyperion.utils.manager_utils import MANAGER_TOKEN

if __name__ == '__main__':
    memory = Memory()
    manager = BaseManager(('', 5602), bytes(MANAGER_TOKEN, encoding='utf8'))
    manager.register('list_indexes', memory.list_indexes)
    manager.register('create_empty_index', memory.create_empty_index)
    manager.register('query_index', memory.query_index)
    manager.register('delete_index', memory.delete_index)
    manager.register('insert_into_index', memory.insert_into_index)
    manager.register('delete_from_index', memory.delete_from_index)
    manager.register('list_documents', memory.list_documents)
    server = manager.get_server()

    ProjectLogger().info('Memory server started.')
    server.serve_forever()
