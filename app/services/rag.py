from pymilvus import Collection
from app.settings import get_settings
from app.db.milvus_client import ensure_collection
import requests

def search_similar(query: str, top_k: int | None = None):
    s = get_settings()
    col: Collection = ensure_collection()

    emb = _embed_one(query)
    col.load()
    res = col.search(
        data=[emb],
        anns_field="embedding",
        param={"metric_type":"COSINE","params":{"nprobe":16}},
        limit=top_k or s.top_k,
        output_fields=["id","title","country","text","price"]
    )
    hits = []
    for hit in res[0]:
        row = dict(hit.entity)
        row["score"] = float(hit.distance)  # con COSINE en Milvus: distancia menor => más parecido
        hits.append(row)
    return hits

def _embed_one(text: str) -> list[float]:
    s = get_settings()
    r = requests.post(f"{s.ollama_host}/api/embeddings", json={"model": s.embed_model, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def build_prompt(question: str, evidence: list[dict]) -> str:
    if not evidence:
        return f"Pregunta: {question}\nNo hay evidencia. Responde: 'no data'."
    ctx = "\n".join([f"- {e['title']} ({e.get('country','')}) precio={e.get('price','?')}\n{e.get('text','')[:500]}" for e in evidence])
    return (
        "Responde SOLO con base en la evidencia; si no alcanza, responde 'no data'.\n"
        f"Pregunta: {question}\n\nEvidencia:\n{ctx}\n\nRespuesta:"
    )

def generate_answer(prompt: str) -> str:
    s = get_settings()
    r = requests.post(f"{s.ollama_host}/api/generate",
                      json={"model": s.gen_model, "prompt": prompt, "stream": False})
    r.raise_for_status()
    return r.json()["response"]
