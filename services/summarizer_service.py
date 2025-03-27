from llama_index.core import SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import settings

class SummarizerService:
    def __init__(self):
        self.llm = Groq(model=settings.LLM_MODEL)
        self.embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)

    def summarize(self, user_input: str) -> str:
        if not user_input.strip():
            return "[Empty input]"

        document = Document(text=user_input)
        splitter = SentenceSplitter(chunk_size=16384)
        nodes = splitter.get_nodes_from_documents([document])

        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True
        )

        query_prompt = '''Please summarize the following as if it was a test paper:
Essay Title:
Author:
Author's Salient Points:
Author's Weak Points:
Points for Improvement:
Score out of 10 (?/10):'''

        response = summary_query_engine.query(query_prompt)
        return str(response)