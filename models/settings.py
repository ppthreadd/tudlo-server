from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Tudlo Essay Summarizer with RAG + VectorDB"
    APP_DESCRIPTION: str = "AI-powered essay summarization service"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM and VectorDB settings
    GROQ_API_KEY: str
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama-3.1-8b-instant"

    # Explicitly disable OpenAI
    DISABLE_OPENAI: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'forbid'  # Prevent extra fields


settings = Settings()