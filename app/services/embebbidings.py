import requests
from app.settings import get_settings

def _embed_one(text: str) -> list[float]:
    s = get_settings()
    r = requests.post(f"{s.ollama_host}/api/embeddings", json={"model": s.embed_model, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def embed_many(texts: list[str]) -> list[list[float]]:
    return [_embed_one(t) for t in texts]
