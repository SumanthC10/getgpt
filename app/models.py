from pydantic import BaseModel
from typing import List

class CompletionResp(BaseModel):
    disease: str
    score: float

class EfoSearchResult(BaseModel):
    efo_id: str
    label: str
    score: float

class EfoSearchResponse(BaseModel):
    results: List[EfoSearchResult]