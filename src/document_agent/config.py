from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"

    data_dir: Path = PROJECT_DIR / ".data"

    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DA_", extra="ignore")


settings = Settings()
