# backend/schemas.py

from pydantic import BaseModel
from typing import Dict, Any


class Document(BaseModel):
    document_id: str
    text: str
    metadata: Dict[str, Any]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]