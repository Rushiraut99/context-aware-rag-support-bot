from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    detail: str

class IngestRequest(BaseModel):
    """
    Request body for /ingest endpoint.
    For simplicity, we ingest all files from data/sample_docs,
    but this can later accept raw text or uploaded files.
    """
    rebuild: bool = False

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    debug: bool = False

class Chunk(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    context: Optional[List[Chunk]] = None
