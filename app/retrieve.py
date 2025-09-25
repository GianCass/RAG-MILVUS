# retrieve.py
from typing import List, Dict, Optional, Literal
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from statistics import mean
import torch
import re, json

COL = "retail_products"
EMB = "intfloat/multilingual-e5-base"

# --- Config de búsqueda ---
SIM_TH = 0.40   # umbral de similitud (IP: 0..1). Ajusta si hace falta
TOPK   = 50     # máximo de resultados a considerar

# --- Carga perezosa del modelo (evita duplicar RAM con --reload) ---
_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embeddings] usando device={device}")
        _model = SentenceTransformer(EMB, device=device)
    return _model

# --- Utilidades ---
def sanitize(text: str) -> str:
    text = text or ""
    text = re.sub(r"(?i)(ignore|override|disregard).*", "", text)
    return text.replace("```", "").strip()

def build_expr(filters: Optional[Dict]) -> Optional[str]:
    """
    Construye una expresión de filtro de Milvus basada en igualdad exacta.
    Ejemplo: {"country":"CO","store":"Exito"} -> country == "CO" and store == "Exito"
    """
    if not filters:
        return None
    parts = []
    for k, v in filters.items():
        if isinstance(v, (int, float)):
            parts.append(f"{k} == {v}")
        else:
            parts.append(f"{k} == {json.dumps(str(v))}")  # escapa strings correctamente
    return " and ".join(parts) if parts else None

# --- BÚSQUEDA SEMÁNTICA (para preguntas tipo "¿cuánto cuesta ...?") ---
def retrieve(question: str, filters: Optional[Dict]=None, topk: int = TOPK, sim_th: float = SIM_TH) -> List[Dict]:
    """
    Retorna hits con campos estructurados (+ score) usando búsqueda vectorial.
    """
    connections.connect(alias="default", host="127.0.0.1", port="19530")
    col = Collection(COL)
    col.load()  # bloqueante

    expr = build_expr(filters)
    qvec = _get_model().encode(["query: " + question], normalize_embeddings=True)

    res = col.search(
        data=qvec,
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},  # HNSW/IP según tu create_collection.py
        limit=topk,
        expr=expr,
        output_fields=[
            "product_id","name","brand","category","store","country",
            "price","unit","size","currency","url","canonical_text"
        ],
    )

    hits: List[Dict] = []
    for hit in res[0]:
        # En IP (inner product) mayor = más similar. Filtramos por umbral.
        if hit.distance < sim_th:
            continue
        e = hit.entity
        hits.append({
            "score": float(hit.distance),
            "product_id": e.get("product_id"),
            "name": e.get("name"),
            "brand": e.get("brand"),
            "category": e.get("category"),
            "store": e.get("store"),
            "country": e.get("country"),
            "price": float(e.get("price")),
            "unit": e.get("unit"),
            "size": float(e.get("size")),
            "currency": e.get("currency"),
            "url": e.get("url"),
            "canonical_text": sanitize(e.get("canonical_text")),
        })
    return hits

# --- LISTADOS DIRECTOS (para "dame todos los de Colombia/Éxito/...") ---
def list_by_filter(filters: Optional[Dict]=None, limit: int=100) -> List[Dict]:
    """
    Consulta estructurada (sin LLM) usando query por filtros exactos.
    OJO: expr vacío devuelve todo; deja un límite razonable.
    """
    connections.connect(alias="default", host="127.0.0.1", port="19530")
    col = Collection(COL)
    col.load()

    expr = build_expr(filters)
    rows = col.query(
        expr=expr or "",
        output_fields=[
            "product_id","name","brand","category","store","country",
            "price","unit","size","currency","url","canonical_text"
        ],
        limit=max(1, min(limit, 1000)),  # tope sano
    )
    # Normaliza algunos tipos/strings
    for r in rows:
        r["price"] = float(r["price"])
        r["size"] = float(r["size"])
        r["canonical_text"] = sanitize(r.get("canonical_text"))
    return rows

# --- AGREGACIONES SIMPLES (min/máx/promedio) ---
def aggregate_prices(filters: Optional[Dict]=None, by: Optional[Literal["store","category","country"]]=None) -> Dict:
    """
    Calcula min/máx/avg de price, global o agrupado por 'store'/'category'/'country'.
    Se hace en Python sobre el resultado de list_by_filter (hasta 1000 filas).
    """
    items = list_by_filter(filters, limit=1000)
    if not items:
        return {"groups": [], "total": 0}

    if not by:
        prices = [x["price"] for x in items]
        return {
            "groups": [{"key": "ALL", "count": len(prices), "min": min(prices), "max": max(prices), "avg": mean(prices)}],
            "total": len(items)
        }

    groups: Dict[str, List[float]] = {}
    for x in items:
        key = x.get(by) or "N/A"
        groups.setdefault(key, []).append(x["price"])

    out = []
    for k, arr in groups.items():
        out.append({
            "key": k,
            "count": len(arr),
            "min": min(arr),
            "max": max(arr),
            "avg": mean(arr)
        })
    return {"groups": out, "total": len(items)}

# --- Test local rápido ---
if __name__ == "__main__":
    print(retrieve("precio del arroz la merced 900g", {"country":"CO","store":"Exito"})[:3])
