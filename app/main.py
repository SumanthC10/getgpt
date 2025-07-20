from fastapi import FastAPI, HTTPException, Query, Body
from .services import (
    semantic_search_efo,
    get_genes_from_source,
    run_pager_analysis,
    get_gene_list
)
from .models import (
    EfoSearchResponse,
    GeneListResponse,
    PagerResponse,
    DiseaseInput,
    GeneInput,
    GetListInput
)
from .providers import GWASCatalog, RummaGEO, OpenTargets

app = FastAPI(title="GetGPT API")

# ==============================================================================
# Gene & Pathway Endpoints
# ==============================================================================

@app.post("/v1/get-list", response_model=GeneListResponse)
def get_list_endpoint(payload: GetListInput = Body(...)):
    """
    Main endpoint to retrieve a curated list of genes based on a disease.
    Accepts either a direct EFO ID or a raw text query.
    """
    try:
        results = get_gene_list(disease_id=payload.disease_id, query=payload.query)
        if not results["results"]:
            raise HTTPException(status_code=404, detail="No genes found for the given input.")
        return results
    except ValueError as e:
        # Catches validation errors from the Pydantic model
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catches errors from the service layer, e.g., query resolution failure
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/v1/genes/gwascatalog", response_model=GeneListResponse)
def gwas_genes_endpoint(payload: DiseaseInput = Body(...)):
    """Retrieves genes from the GWAS Catalog for a given EFO ID."""
    results = get_genes_from_source(GWASCatalog, payload.disease_id)
    return {"results": results}

@app.post("/v1/genes/rummageo", response_model=GeneListResponse)
def rummageo_genes_endpoint(payload: DiseaseInput = Body(...)):
    """Retrieves genes from RummaGEO for a given EFO ID."""
    results = get_genes_from_source(RummaGEO, payload.disease_id)
    return {"results": results}

@app.post("/v1/genes/opentargets", response_model=GeneListResponse)
def opentargets_genes_endpoint(payload: DiseaseInput = Body(...)):
    """Retrieve genes from OpenTargets for a given EFO ID."""
    results = get_genes_from_source(OpenTargets, payload.disease_id)
    return {"results": results}

@app.post("/v1/genes/pager", response_model=PagerResponse)
def pager_analysis_endpoint(payload: GeneInput = Body(...)):
    """Runs PAGER pathway analysis on a provided list of genes."""
    results = run_pager_analysis(payload.gene_list)
    return {"results": results}

# ==============================================================================
# Existing EFO Search Endpoint
# ==============================================================================

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