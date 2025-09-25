# ingest.py — Ingesta a Milvus con esquema "completo" (14 campos)
# Campos: product_id, name, brand, category, store, country, price, unit,
#         size, currency, last_seen, url, canonical_text, embedding

import os, csv, math, time, argparse, itertools
from typing import Dict, List, Tuple

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)

# ========= Config desde .env =========
MILVUS_HOST       = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT       = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "retail_products")

# Embeddings: BACKEND puede ser 'ollama' o 'hf'
EMBED_BACKEND     = os.getenv("EMBED_BACKEND", "hf").lower()
EMBED_MODEL       = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# ========= Utilidades =========
FIELD_ORDER = [
    "product_id", "name", "brand", "category", "store", "country",
    "price", "unit", "size", "currency", "last_seen", "url",
    "canonical_text", "embedding"
]

def canonical(r: Dict) -> str:
    """Texto canónico compacto/consistente para embeddings."""
    return (
        f"{r['name']}. Marca: {r['brand']}. Categoría: {r['category']}. "
        f"Presentación: {r['size']}{r['unit']}. Precio: {r['price']} {r['currency']}. "
        f"Tienda: {r['store']}. País: {r['country']}."
    )

def l2_normalize(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / s for x in vec]

def ensure_connection():
    connections.connect(alias="default", host=MILVUS_HOST, port=str(MILVUS_PORT))

def ensure_collection(dim: int) -> Collection:
    """Crea la colección con 14 campos si no existe. Si existe, valida campos y dimensión."""
    name = MILVUS_COLLECTION
    if not utility.has_collection(name):
        fields = [
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
            FieldSchema(name="name",        dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="brand",       dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="category",    dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="store",       dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="country",     dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="price",       dtype=DataType.DOUBLE),
            FieldSchema(name="unit",        dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="size",        dtype=DataType.DOUBLE),
            FieldSchema(name="currency",    dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="last_seen",   dtype=DataType.INT64),
            FieldSchema(name="url",         dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="canonical_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description=f"Retail products (14 campos) | embedder={EMBED_BACKEND}:{EMBED_MODEL}")
        col = Collection(name=name, schema=schema)
        col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}})
        return col

    # Si ya existe, validar que tenga los campos esperados y la dimensión
    col = Collection(name)
    existing = {f.name: f for f in col.schema.fields}
    for fname in FIELD_ORDER[:-1]:  # excepto embedding que validamos abajo
        if fname not in existing:
            raise ValueError(f"El campo '{fname}' no existe en la colección '{name}'. Esquema incompatible.")
    # Validar dim de embedding
    emb_field = existing.get("embedding")
    emb_dim = None
    try:
        # Algunas versiones exponen params con 'dim'
        if hasattr(emb_field, "params"):
            emb_dim = emb_field.params.get("dim")
    except Exception:
        emb_dim = None
    if emb_dim and int(emb_dim) != int(dim):
        raise ValueError(f"Dimensión de embedding incompatible. Colección={emb_dim}, nuevo={dim}")
    return col

# ========= Backends de embeddings =========
def embed_ollama(texts: List[str]) -> List[List[float]]:
    import requests
    out = []
    url = f"{OLLAMA_HOST}/api/embeddings"
    for t in texts:
        r = requests.post(url, json={"model": EMBED_MODEL, "prompt": t})
        r.raise_for_status()
        vec = r.json()["embedding"]
        out.append(l2_normalize(vec))
    return out

def embed_hf(texts: List[str]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    prepped = [("passage: " + t) if "e5" in EMBED_MODEL else t for t in texts]
    vecs = model.encode(prepped, normalize_embeddings=True)
    return [v.tolist() for v in vecs]

def embed_texts(texts: List[str]) -> Tuple[List[List[float]], int]:
    if EMBED_BACKEND == "ollama":
        vecs = embed_ollama(texts)
    else:
        vecs = embed_hf(texts)
    dim = len(vecs[0]) if vecs else 0
    return vecs, dim

# ========= I/O CSV =========
def read_csv_rows(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # casting / defaults
            r["last_seen"] = int(r.get("last_seen") or int(time.time() * 1000))
            r["price"] = float(r["price"]) if r.get("price") not in (None, "",) else 0.0
            r["size"] = float(r["size"]) if r.get("size") not in (None, "",) else 0.0
            # canonical_text si no viene
            r["canonical_text"] = r.get("canonical_text") or canonical(r)
            rows.append(r)
    return rows

# ========= Deduplicación =========
def delete_existing_ids(col: Collection, ids: List[str], batch: int = 500):
    """Elimina por lotes registros con product_id ya existentes (cuando no hay upsert)."""
    col.load()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        # Escapar comillas y formar lista para expr
        escaped = [f'"{x}"' for x in chunk]
        expr = f"product_id in [{', '.join(escaped)}]"
        col.delete(expr)

# ========= Inserción / Upsert =========
def to_data_lists(rows: List[Dict], vecs: List[List[float]]) -> List[list]:
    return [
        [r["product_id"] for r in rows],
        [r["name"] for r in rows],
        [r["brand"] for r in rows],
        [r["category"] for r in rows],
        [r["store"] for r in rows],
        [r["country"] for r in rows],
        [float(r.get("price") or 0.0) for r in rows],
        [r["unit"] for r in rows],
        [float(r.get("size") or 0.0) for r in rows],
        [r["currency"] for r in rows],
        [int(r.get("last_seen")) for r in rows],
        [r["url"] for r in rows],
        [r["canonical_text"] for r in rows],
        vecs,
    ]

def insert_batches(col: Collection, rows: List[Dict], vecs: List[List[float]], batch_size: int = 512):
    assert len(rows) == len(vecs)
    col.load()
    total = 0
    has_upsert = hasattr(col, "upsert")
    for i in range(0, len(rows), batch_size):
        chunk_r = rows[i:i+batch_size]
        chunk_v = vecs[i:i+batch_size]
        data = to_data_lists(chunk_r, chunk_v)

        if has_upsert:
            col.upsert(data)
        else:
            # borrar posibles duplicados y luego insert
            delete_existing_ids(col, [r["product_id"] for r in chunk_r])
            col.insert(data)
        total += len(chunk_r)
    col.flush()
    return total

# ========= Main =========
def main():
    parser = argparse.ArgumentParser(description="Ingesta CSV -> Milvus (14 campos)")
    parser.add_argument("--csv", type=str, default="data/sample.csv", help="Ruta del CSV")
    args = parser.parse_args()

    csv_path = os.path.normpath(args.csv)
    if not os.path.exists(csv_path):
        alt = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sample.csv"))
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"No existe el CSV: {args.csv}")

    print(f"[CFG] Milvus: {MILVUS_HOST}:{MILVUS_PORT} | Col: {MILVUS_COLLECTION}")
    print(f"[CFG] Embeddings: backend={EMBED_BACKEND} model={EMBED_MODEL}")
    print(f"[CSV] {csv_path}")

    rows = read_csv_rows(csv_path)
    if not rows:
        print("No se leyeron filas del CSV. Revisa el archivo.")
        return

    texts = [r["canonical_text"] for r in rows]
    print(f"[EMB] Generando embeddings para {len(texts)} items…")
    vecs, dim = embed_texts(texts)
    if dim <= 0:
        raise RuntimeError("No se obtuvo dimensión de embeddings.")

    print(f"[MILVUS] Conectando y asegurando colección (dim={dim})…")
    ensure_connection()
    col = ensure_collection(dim)

    print("[MILVUS] Insertando por lotes…")
    total = insert_batches(col, rows, vecs, batch_size=512)
    print(f"[OK] Ingestados {total} registros en '{MILVUS_COLLECTION}'.")

if __name__ == "__main__":
    main()
