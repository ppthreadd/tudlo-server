from llama_index.core import SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models.settings import settings
import logging


class SummarizerService:
    def __init__(self):
        try:
            # Configure global LlamaIndex settings
            Settings.llm = Groq(
                model=settings.LLM_MODEL,
                api_key=settings.GROQ_API_KEY
            )
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=settings.EMBEDDING_MODEL
            )

            self.splitter = SentenceSplitter(chunk_size=16384)
            logging.info("SummarizerService initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize SummarizerService: {str(e)}")
            raise

    def summarize(self, user_input: str) -> str:
        """Summarize text using Groq LLM with test paper format"""
        if not user_input.strip():
            return "[Empty input]"

        try:
            # Create document and nodes
            document = Document(text=user_input)
            nodes = self.splitter.get_nodes_from_documents([document])

            # Configure and query the summary engine
            summary_index = SummaryIndex(nodes)
            query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True
            )

            prompt = """Please summarize as a test paper:
            Essay Title:
            Author:
            Key Points (3-5):
            Weaknesses (2-3):
            Improvements Suggested:
            Score (1-10):"""

            response = query_engine.query(prompt)
            return str(response)

        except Exception as e:
            logging.error(f"Summarization failed: {str(e)}")
            return f"Summarization error: {str(e)}"


# Initialize service instance
summarizer_service = SummarizerService()