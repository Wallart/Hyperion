from multiprocessing import Lock
from hyperion.analysis import load_file
from hyperion.utils import ProjectPaths
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, GPTVectorStoreIndex

import os


class Memory:

    def __init__(self):
        self._index = {}
        self._lock = Lock()
        self._indexes_dir = ProjectPaths().resources_dir / 'indexes'
        os.environ['OPENAI_API_KEY'] = load_file(ProjectPaths().resources_dir / 'keys' / 'openai_api.key')[0]

    def _initialize_index(self, index_name, document):
        with self._lock:
            persist_dir = self._indexes_dir / index_name

            if index_name not in self._index and not persist_dir.exists():
                storage_context = StorageContext.from_defaults()
                self._index[index_name] = GPTVectorStoreIndex.from_documents([document], storage_context=storage_context)
                self._index[index_name].index_struct.index_id = index_name
                storage_context.persist(persist_dir)
            else:
                if index_name not in self._index:
                    self._index[index_name] = load_index_from_storage(StorageContext.from_defaults())

                self._index[index_name].insert(document)
                self._index[index_name].storage_context.persist(persist_dir)

    def insert_into_index(self, index_name, document_path, doc_id=None):
        document = SimpleDirectoryReader(input_files=[document_path]).load_data()[0]
        if doc_id is not None:
            document.doc_id = doc_id

        self._initialize_index(index_name, document)

    def query_index(self, index_name, query_text):
        with self._lock:
            query_engine = self._index[index_name].as_query_engine()
            response = query_engine.query(query_text)
        return str(response)
