from multiprocessing.managers import BaseManager
from hyperion.utils.manager_utils import MANAGER_TOKEN

if __name__ == '__main__':
    memoryManager = BaseManager(('', 5602), bytes(MANAGER_TOKEN, encoding='utf8'))
    memoryManager.register('list_indexes')
    memoryManager.register('create_empty_index')
    memoryManager.register('query_index')
    memoryManager.register('delete_index')
    memoryManager.register('insert_into_index')
    memoryManager.register('delete_from_index')
    memoryManager.register('list_documents')
    memoryManager.connect()

    index = 'hyperion'

    indexes = memoryManager.list_indexes(index)
    docs = memoryManager.list_documents(index)._getvalue()
    # memoryManager.delete_index(index)
    if len(docs) > 0:
        memoryManager.delete_from_index(index, docs[0])
