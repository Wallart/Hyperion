from typing import List
from pathlib import Path
from llama_index.llms import OpenAI
from hyperion.utils.paths import ProjectPaths
from multiprocessing import Lock, Manager, Process
from hyperion.utils import load_file, ProjectLogger
from hyperion.utils.memory_utils import ConcurrentDict
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, ServiceContext

import os
import shutil


class Memory:

    def __init__(self, llama_host, llama_port):
        self._index = ConcurrentDict()
        self._state = Manager().Value('c', '')  # Manager is concurrent access safe
        self._indexes_dir = ProjectPaths().cache_dir / 'indexes'

        llama_cpp_url = f'http://{llama_host}:{llama_port}'
        api_key = os.environ['OPENAI_API'] if 'OPENAI_API' in os.environ else load_file(ProjectPaths().resources_dir / 'keys' / 'openai_api.key')[0]
        open_ai = ServiceContext.from_defaults(
            llm=OpenAI(api_key=api_key),
            embed_model=OpenAIEmbedding(api_key=api_key)
        )
        local = ServiceContext.from_defaults(
            llm=OpenAI(api_key='sk-no-key-required', api_base=f'{llama_cpp_url}/v1'),
            embed_model=OpenAIEmbedding(api_key='sk-no-key-required', api_base=llama_cpp_url)
        )
        self._service_ctx = dict(openai=open_ai, local=local)

    def _get_context(self, llm_name):
        ctx_name = 'openai' if llm_name in ['', None] or llm_name.startswith('gpt') else 'local'

        ProjectLogger().debug(f'Loading {ctx_name} as service context')
        return self._service_ctx[ctx_name]

    def _reset_state(self):
        self._state.value = ''

    # def _load_all_indexes(self):
    #     indexes = self.list_indexes()
    #     _ = [self._load_index(index) for index in indexes]

    def _load_index(self, index, llm=None):
        index_path = self._indexes_dir / index
        if index not in self._index and index_path.is_dir():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=index_path)
                self._index[index] = [Lock(), load_index_from_storage(storage_context, service_context=self._get_context(llm))]
            except Exception as e:
                ProjectLogger().warning(e)

    def _initialize_index(self, index_name, document, llm):
        self.set_status('indexing')

        self._load_index(index_name)
        if index_name in self._index:
            with self._index[index_name][0]:
                self._index[index_name][1].insert(document)
        else:
            # Nothing found during lazy loading. Create a new index from this document
            self._index[index_name] = [Lock(), VectorStoreIndex.from_documents([document], service_context=self._get_context(llm))]

        # Necessary even with insertion
        with self._index[index_name][0]:
            self._index[index_name][1].storage_context.persist(self._indexes_dir / index_name)

        self._reset_state()

    def get_status(self):
        return self._state.value

    def set_status(self, state):
        self._state.value = state

    def create_empty_index(self, index):
        ProjectLogger().info(f'Creating empty index {index}')
        index_path = self._indexes_dir / index
        os.makedirs(index_path, exist_ok=True)

    def list_indexes(self):
        ProjectLogger().info('Listing indexes')
        indexes = [index.name for index in self._indexes_dir.glob('*') if (self._indexes_dir / index).is_dir()]
        return indexes

    def list_documents(self, index_name, llm) -> List[str]:
        ProjectLogger().info(f'Listing documents in index {index_name}')
        docs = []
        try:
            self._load_index(index_name)
            if index_name in self._index:
                with self._index[index_name][0]:
                    storage_context = StorageContext.from_defaults(persist_dir=self._indexes_dir / index_name)
                    self._index[index_name][1] = load_index_from_storage(storage_context, service_context=self._get_context(llm))
                    docs = self._index[index_name][1].ref_doc_info.keys()
        except Exception as e:
            ProjectLogger().warning(e)

        return docs

    def async_insert_into_index(self, *args, **kwargs):
        p = Process(target=self.insert_into_index, args=args, kwargs=kwargs)
        p.start()

    def insert_into_index(self, index_name, doc_id, document_path, llm):
        ProjectLogger().info(f'Inserting {doc_id} to index {index_name}')
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        for i, document in enumerate(documents, start=1):
            if len(documents) > 1:
                new_doc_id = list(os.path.splitext(doc_id))
                new_doc_id.append(f' [page {i}]')
                document.doc_id = ''.join(new_doc_id)
            else:
                document.doc_id = doc_id
            self._initialize_index(index_name, document, llm)

        Path(document_path).unlink(missing_ok=True)
        ProjectLogger().info(f' {doc_id} insertion to index {index_name} completed.')

    def delete_from_index(self, index_name, doc_id):
        ProjectLogger().info(f'Deleting {doc_id} from index {index_name}')
        self._load_index(index_name)
        if index_name in self._index:
            with self._index[index_name][0]:
                self._index[index_name][1].delete_ref_doc(doc_id, delete_from_docstore=True)
                self._index[index_name][1].storage_context.persist(self._indexes_dir / index_name)

    def query_index(self, index_name, query_text, llm):
        ProjectLogger().info(f'Querying {index_name} with {query_text}')
        self._load_index(index_name)
        if index_name in self._index:
            with self._index[index_name][0]:
                query_engine = self._index[index_name][1].as_query_engine(service_context=self._get_context(llm))
                response = query_engine.query(query_text)
            return str(response)

    def delete_index(self, index_name):
        ProjectLogger().info(f'Deleting {index_name}')
        try:
            self._load_index(index_name)
            if index_name in self._index:
                del self._index[index_name]
        except Exception as e:
            ProjectLogger().warning(e)
        index_path = self._indexes_dir / index_name
        shutil.rmtree(index_path, ignore_errors=True)
