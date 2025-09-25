from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

COL = "retail_products"

def main():
    connections.connect(alias="default", host="127.0.0.1", port="19530")

    fields = [
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="name",       dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="brand",      dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="category",   dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="store",      dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="country",    dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="price",      dtype=DataType.FLOAT),
        FieldSchema(name="unit",       dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="size",       dtype=DataType.FLOAT),
        FieldSchema(name="currency",   dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="last_seen",  dtype=DataType.INT64),
        FieldSchema(name="url",        dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="canonical_text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="vector",     dtype=DataType.FLOAT_VECTOR, dim=768),
    ]
    schema = CollectionSchema(fields, description="Retail products for RAG (no hallucinations)")

    if not utility.has_collection(COL):
        col = Collection(name=COL, schema=schema)
        index_params = {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 32, "efConstruction": 200}}
        col.create_index(field_name="vector", index_params=index_params)
        print(f"Collection {COL} creada e indexada.")
    else:
        print(f"La colección {COL} ya existe.")

if __name__ == "__main__":
    main()
