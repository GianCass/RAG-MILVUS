# app/settings.py — Pydantic v2 compatible
from functools import lru_cache
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ⚡️ CORS: lista de orígenes permitidos (separados por coma en .env)
    cors_origins: List[str] = Field(
        default=["http://localhost:4200", "http://localhost:5173"],
        alias="CORS_ORIGINS"
    )

    # Operación
    enable_admin_routes: bool = Field(default=False, alias="ENABLE_ADMIN_ROUTES")

    # Milvus
    milvus_host: str = Field(default="127.0.0.1", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="retail_products", alias="MILVUS_COLLECTION")
    milvus_dim: int = Field(default=768, alias="MILVUS_DIM")

    # Modelos (Ollama / Embeddings)
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")
    embed_backend: str = Field(default="hf", alias="EMBED_BACKEND")
    embed_model: str = Field(default="intfloat/multilingual-e5-base", alias="EMBED_MODEL")
    gen_model: str = Field(default="phi3:mini", alias="GEN_MODEL")
    abstain_threshold: float = Field(default=0.35, alias="ABSTAIN_THRESHOLD")
    top_k: int = Field(default=5, alias="TOP_K")

    # Busca .env en app/.env y en la raíz ../.env
    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        case_sensitive=False,
        extra="ignore",
    )

    # --- Normaliza CORS_ORIGINS de str → list ---
    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

@lru_cache
def get_settings() -> Settings:
    return Settings()
