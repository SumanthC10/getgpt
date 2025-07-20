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
from .services import semantic_search_efo
from .models import EfoSearchResponse

@app.get("/v1/efo_search", response_model=EfoSearchResponse)
def efo_search_endpoint(
    q: str = Query(..., min_length=3, description="Query string to search for"),
    top_k: int = Query(5, description="Number of results to return. Use -1 for all.")
):
    try:
        results = semantic_search_efo(q, top_k)
        return {"results": results}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))