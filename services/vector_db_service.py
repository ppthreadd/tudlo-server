import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader
from config import settings


class VectorDBService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_or_create_collection("tudlo_docs")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

    def load_documents(self, docs_path: str = "./docs"):
        documents = SimpleDirectoryReader(docs_path).load_data()
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        return vector_index.as_query_engine(similarity_top_k=3)