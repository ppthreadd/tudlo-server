from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import settings

def get_llm():
    return Groq(model=settings.LLM_MODEL)

def get_embed_model():
    return HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)