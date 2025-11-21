from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Central place for configuration.
    Values can come from environment variables or a .env file.
    """
    openai_api_key: str
    model_name: str = "gpt-4o-mini"  # You can change this if needed
    embedding_model_name: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
