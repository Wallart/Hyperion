from typing import List
from pathlib import Path
from multiprocessing import Lock
from hyperion.utils import load_file, ProjectLogger
from hyperion.utils.paths import ProjectPaths
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex

import os
import shutil


class Memory:

    def __init__(self):
        self._index = {}
        self._lock = Lock()
        self._indexes_dir = ProjectPaths().resources_dir / 'indexes'
        os.environ['OPENAI_API_KEY'] = load_file(ProjectPaths().resources_dir / 'keys' / 'openai_api.key')[0]

    def _load_all_indexes(self):
        indexes = self.list_indexes()
        _ = [self._load_index(index) for index in indexes]

    def _load_index(self, index):
        index_path = self._indexes_dir / index
        if index not in self._index and index_path.is_dir():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=index_path)
                self._index[index] = load_index_from_storage(storage_context)
            except Exception:
                pass

    def _initialize_index(self, index_name, document):
        with self._lock:
            self._load_index(index_name)
            # Nothing found during lazy loading. Create a new index from this document
            if index_name in self._index:
                self._index[index_name].insert(document)
            else:
                self._index[index_name] = VectorStoreIndex.from_documents([document])
                self._index[index_name].storage_context.persist(self._indexes_dir / index_name)

    def create_empty_index(self, index):
        ProjectLogger().info(f'Creating empty index {index}')
        with self._lock:
            index_path = self._indexes_dir / index
            index_path.mkdir(exist_ok=True)

    def list_indexes(self):
        ProjectLogger().info('Listing indexes')
        with self._lock:
            return [index.name for index in self._indexes_dir.glob('*') if (self._indexes_dir / index).is_dir()]

    def list_documents(self, index_name) -> List[str]:
        ProjectLogger().info(f'Listing documents in index {index_name}')
        docs = []
        with self._lock:
            try:
                self._load_index(index_name)
                if index_name in self._index:
                    doc_infos = self._index[index_name].ref_doc_info.values()
                    docs = [Path(e.metadata['file_path']).name for e in doc_infos]
            except Exception:
                pass

        return docs

    def insert_into_index(self, index_name, doc_id, document_path):
        ProjectLogger().info(f'Inserting {doc_id} to index {index_name}')
        document = SimpleDirectoryReader(input_files=[document_path]).load_data()[0]
        document.doc_id = doc_id

        self._initialize_index(index_name, document)

    def delete_from_index(self, index_name, doc_id):
        ProjectLogger().info(f'Deleting {doc_id} from index {index_name}')
        with self._lock:
            self._load_index(index_name)
            if index_name in self._index:
                self._index[index_name].delete_ref_doc(doc_id, delete_from_docstore=True)

    def query_index(self, index_name, query_text):
        ProjectLogger().info(f'Querying {index_name} with {query_text}')
        with self._lock:
            self._load_index(index_name)
            if index_name in self._index:
                query_engine = self._index[index_name].as_query_engine()
                response = query_engine.query(query_text)
                return str(response)

    def delete_index(self, index_name):
        ProjectLogger().info(f'Deleting {index_name}')
        with self._lock:
            try:
                self._load_index(index_name)
                if index_name in self._index:
                    del self._index[index_name]
            except Exception:
                pass
            index_path = self._indexes_dir / index_name
            shutil.rmtree(index_path, ignore_errors=True)
