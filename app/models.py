from pydantic import BaseModel, Field, model_validator
from typing import List, Any, Dict, Optional

# ==============================================================================
# Request Models
# ==============================================================================

class DiseaseInput(BaseModel):
    disease_id: str = Field(..., description="An EFO identifier for a disease, e.g., 'EFO_0000270' or 'EFO:0000270'.")

class GeneInput(BaseModel):
    gene_list: List[str] = Field(..., description="A list of gene symbols to be analyzed.", min_items=1)

class GetListInput(BaseModel):
    disease_id: Optional[str] = Field(default=None, description="An EFO identifier for a disease, e.g., 'EFO_0000270' or 'EFO:0000270'.")
    query: Optional[str] = Field(default=None, description="A raw text query for a disease, e.g., 'Alzheimer's disease'.")

    @model_validator(mode='before')
    @classmethod
    def check_one_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            has_disease_id = 'disease_id' in data and data.get('disease_id')
            has_query = 'query' in data and data.get('query')
            if has_disease_id and has_query:
                raise ValueError('Provide either disease_id or query, not both.')
            if not has_disease_id and not has_query:
                raise ValueError('Either disease_id or query must be provided.')
        return data

# ==============================================================================
# Gene & Data Source Response Models
# ==============================================================================

class Gene(BaseModel):
    gene_symbol: str
    source: List[str]  # Changed from str to List[str]
    score: float
    evidence: Any  # Can be a list of strings, dicts, or empty

class GeneListResponse(BaseModel):
    results: List[Gene]

# ==============================================================================
# PAGER Pathway Analysis Response Models
# ==============================================================================

class PagerEntity(BaseModel):
    pag_id: str = Field(..., alias="PAG ID")
    pag_name: str = Field(..., alias="PAG Name")
    gs_size: int = Field(..., alias="GS Size")
    overlap: int = Field(..., alias="Overlap")
    p_value: float = Field(..., alias="P-value")
    fdr: float = Field(..., alias="FDR")

    class Config:
        populate_by_name = True # Allows using aliases for field names

class PagerResponse(BaseModel):
    results: List[PagerEntity]

# ==============================================================================
# EFO Search Response Models
# ==============================================================================

class EfoSearchResult(BaseModel):
    efo_id: str
    label: str
    score: float

class EfoSearchResponse(BaseModel):
    results: List[EfoSearchResult]

# ==============================================================================
# Deprecated Models
# ==============================================================================

class CompletionResp(BaseModel):
    disease: str
    score: float