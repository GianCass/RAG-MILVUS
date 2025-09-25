from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from app.settings import get_settings

def connect():
    s = get_settings()
    connections.connect(alias="default", host=s.milvus_host, port=str(s.milvus_port))

def ensure_collection() -> Collection:
    s = get_settings()
    connect()
    name = s.milvus_collection
    if not utility.has_collection(name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="price", dtype=DataType.DOUBLE),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=get_settings().milvus_dim),
        ]
        schema = CollectionSchema(fields, description="Retail products")
        col = Collection(name=name, schema=schema)
        col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}})
    return Collection(name)

def query_rows(country: str | None = None, offset: int = 0, limit: int = 50):
    """Listado simple con filtro por país y paginación."""
    col = ensure_collection()
    col.load()
    expr = None
    if country:
        # NOTA: para VARCHAR en Milvus, usa comillas dobles
        expr = f'country == "{country}"'
    fields = ["id", "title", "country", "price"]
    res = col.query(expr=expr, output_fields=fields, offset=offset, limit=limit)
    return res
