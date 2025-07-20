from pydantic import BaseModel

class CompletionResp(BaseModel):
    disease: str
    score: float