from hyperion.pipelines.memory import Memory
from hyperion.utils.logger import ProjectLogger
from multiprocessing.managers import BaseManager

if __name__ == '__main__':
    memory = Memory()
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_index', memory.query_index)
    manager.register('insert_into_index', memory.insert_into_index)
    # manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    ProjectLogger().info('Memory server started.')
    server.serve_forever()
