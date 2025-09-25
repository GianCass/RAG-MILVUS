from pydantic import BaseModel, Field
from typing import List, Optional

class AskRequest(BaseModel):
    query: str
    top_k: int | None = None
    abstain_threshold: float | None = None

class AskResponse(BaseModel):
    answer: str
    hits: List[dict] = Field(default_factory=list)

class ListRequest(BaseModel):
    country: str | None = None
    offset: int = 0
    limit: int = 50

class ListResponse(BaseModel):
    items: List[dict]
    total: int | None = None  # opcional (Milvus no devuelve total exacto sin truco)
