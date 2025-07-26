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
    disease_ids: Optional[List[str]] = Field(default=None, description="A list of EFO identifiers for diseases.")
    query: Optional[str] = Field(default=None, description="A raw text query for a disease.")

    @model_validator(mode='before')
    @classmethod
    def check_one_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            has_disease_ids = 'disease_ids' in data and data.get('disease_ids')
            has_query = 'query' in data and data.get('query')
            if has_disease_ids and has_query:
                raise ValueError('Provide either disease_ids or query, not both.')
            if not has_disease_ids and not has_query:
                raise ValueError('Either disease_ids or query must be provided.')
        return data

# ==============================================================================
# Gene & Data Source Response Models
# ==============================================================================

class SourceInfo(BaseModel):
    name: str
    url: str

class Gene(BaseModel):
    gene_symbol: str
    source: List[SourceInfo]
    overall_score: float = Field(..., description="The combined, weighted score from all available data sources.")
    g_score: float = Field(default=0, description="Score from GWAS Catalog, based on the -log10(p-value).")
    e_score: float = Field(default=0, description="Score from RummaGEO, based on GSEA enrichment p-value.")
    t_score: float = Field(default=0, description="Association score from OpenTargets (0 to 1).")
    evidence: Any

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
# Single-Source Response Models
# ==============================================================================

class GwasGene(BaseModel):
    gene_symbol: str
    source: List[str]
    g_score: float = Field(default=0, description="Score from GWAS Catalog, based on the -log10(p-value).")
    evidence: Any

class GwasGeneResponse(BaseModel):
    results: List[GwasGene]


class RummaGEOGene(BaseModel):
    gene_symbol: str
    source: List[str]
    e_score: float = Field(default=0, description="Score from RummaGEO, based on GSEA enrichment p-value.")
    evidence: Any

class RummaGEOGeneResponse(BaseModel):
    results: List[RummaGEOGene]


class OpenTargetsGene(BaseModel):
    gene_symbol: str
    source: List[str]
    t_score: float = Field(default=0, description="Association score from OpenTargets (0 to 1).")
    evidence: Any

class OpenTargetsGeneResponse(BaseModel):
    results: List[OpenTargetsGene]