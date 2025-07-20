from fastapi import FastAPI, HTTPException, Query
from .services import autocomplete
from .models import CompletionResp

app = FastAPI(title="GetGPT Autocomplete API")

@app.get("/v1/autocomplete", response_model=list[CompletionResp])
def autocomplete_endpoint(
    q: str = Query(..., min_length=2, description="User prefix / query string"),
    top_n: int = Query(8, le=20)
):
    if not q.strip():
        raise HTTPException(400, "Empty query")
    
    return autocomplete(q, top_n)