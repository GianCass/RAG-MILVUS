from fastapi import APIRouter
from app.schemas import ListRequest, ListResponse
from app.db.milvus_client import query_rows

router = APIRouter(prefix="/products", tags=["products"])

@router.post("/list", response_model=ListResponse)
def list_products(payload: ListRequest):
    items = query_rows(country=payload.country, offset=payload.offset, limit=payload.limit)
    return ListResponse(items=items, total=None)
