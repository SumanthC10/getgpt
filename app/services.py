import re
import json
import logging
import pathlib
from typing import List, Set, Dict, Any
from itertools import chain
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

from .providers import get_pager_search_results, GWASCatalog, RummaGEO, OpenTargets

logger = logging.getLogger(__name__)

# ==============================================================================
# Helper Functions
# ==============================================================================

def normalize_efo_id(efo_id: str, source_name: str = None) -> str:
    """
    Normalizes an EFO ID. For RummaGEO, ensures 'EFO:xxxx' format.
    For all other sources, ensures 'EFO_xxxx' format.
    """
    if source_name == "RummaGEO":
        return efo_id.replace("_", ":")
    else:
        return efo_id.replace(":", "_")

# ==============================================================================
# Gene Data Source Services
# ==============================================================================

def get_genes_from_source(source_class, disease_id: str) -> List[Dict[str, Any]]:
    """
    A generic function to instantiate a data source class and get genes.
    It now normalizes the EFO ID before processing.
    """
    try:
        source_instance = source_class()
        normalized_id = normalize_efo_id(disease_id, source_class.__name__)
        df = source_instance.get_genes(normalized_id)
        if df.empty:
            return []
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching data from {source_class.__name__} for {disease_id}: {e}")
        return []

# ==============================================================================
# Main Orchestration Service
# ==============================================================================

def get_gene_list(disease_id: str = None, query: str = None) -> Dict[str, Any]:
    """
    Orchestrates the gene list retrieval pipeline:
    1. If a raw query is provided, use semantic search to find the best EFO ID.
    2. Fetches genes from GWAS, RummaGEO, and OpenTargets concurrently using the EFO ID.
    3. Aggregates and de-duplicates the final gene list.
    """
    from concurrent.futures import ThreadPoolExecutor

    target_efo_id = disease_id
    
    # --- Step 1: Handle raw text query if provided ---
    if query:
        try:
            search_results = semantic_search_efo(query, top_k=1)
            if not search_results:
                raise RuntimeError(f"No EFO term found for query: '{query}'")
            target_efo_id = search_results[0]['efo_id']
            logger.info(f"Query '{query}' mapped to EFO ID: {target_efo_id}")
        except Exception as e:
            logger.error(f"Failed to resolve query '{query}' to an EFO ID: {e}")
            return {"results": []} # Return empty if query fails
 
    if not target_efo_id:
         return {"results": []}

    # --- Step 2: Fetch genes from all sources concurrently ---
    gene_sources = [GWASCatalog, RummaGEO, OpenTargets]
    all_genes = []
    
    with ThreadPoolExecutor(max_workers=len(gene_sources)) as executor:
        future_to_source = {executor.submit(get_genes_from_source, source, target_efo_id): source for source in gene_sources}
        
        for future in future_to_source:
            try:
                genes = future.result()
                if genes:
                    all_genes.extend(genes)
            except Exception as e:
                source_name = future_to_source[future].__name__
                logger.error(f"Error retrieving results from {source_name}: {e}")

    if not all_genes:
        return {"results": []}
 
     # --- Step 3: Aggregate and de-duplicate genes ---
    # --- Step 3: Aggregate and de-duplicate genes ---
    aggregated_genes = {}
    for gene in all_genes:
        symbol = gene['gene_symbol']
        if symbol not in aggregated_genes:
            aggregated_genes[symbol] = {
                'gene_symbol': symbol,
                'source': set(),
                'g_score': 0,
                'e_score': 0,
                't_score': 0,
                'evidence': []
            }
        
        # Aggregate sources and evidence
        aggregated_genes[symbol]['source'].add(gene['source'])
        if gene.get('evidence'):
            aggregated_genes[symbol]['evidence'].extend(gene['evidence'])
        
        # Store individual scores, keeping the max if a gene appears in one source multiple times
        if 'g_score' in gene:
            aggregated_genes[symbol]['g_score'] = max(aggregated_genes[symbol]['g_score'], gene['g_score'])
        if 'e_score' in gene:
            aggregated_genes[symbol]['e_score'] = max(aggregated_genes[symbol]['e_score'], gene['e_score'])
        if 't_score' in gene:
            aggregated_genes[symbol]['t_score'] = max(aggregated_genes[symbol]['t_score'], gene['t_score'])

    # --- Step 4: Calculate overall score and finalize list ---
    final_gene_list = []
    for gene_data in aggregated_genes.values():
        # Define weights
        w_g, w_e, w_t = 1.0, 1.0, 1.0
        
        g = gene_data['g_score']
        e = gene_data['e_score']
        t = gene_data['t_score']
        
        # Calculate weighted average, accounting for missing scores
        numerator = (w_g * g) + (w_e * e) + (w_t * t)
        denominator = w_g * (1 if g > 0 else 0) + w_e * (1 if e > 0 else 0) + w_t * (1 if t > 0 else 0)
        
        gene_data['overall_score'] = numerator / denominator if denominator > 0 else 0
        gene_data['source'] = sorted(list(gene_data['source']))
        final_gene_list.append(gene_data)

    # Sort by overall_score descending
    final_gene_list.sort(key=lambda x: x['overall_score'], reverse=True)
     
    return {"results": final_gene_list}

# ==============================================================================
# PAGER Pathway Analysis Service
# ==============================================================================

def run_pager_analysis(gene_list: List[str]) -> List[Dict[str, Any]]:
    """
    Runs a PAGER pathway analysis on a given gene list and returns the results.
    """
    if not gene_list:
        return []
    
    # Convert gene list to a newline-separated string for PAGER input
    gene_input = "\n".join(gene_list)
    
    try:
        # Call the provider function to get the PAGER results
        pager_df = get_pager_search_results(gene_input)
        
        if pager_df.empty:
            return []
            
        # Convert DataFrame to a list of dictionaries for Pydantic modeling
        return pager_df.to_dict(orient='records')
        
    except Exception as e:
        logger.error(f"An error occurred during PAGER analysis: {e}")
        # In a production scenario, you might want to raise a specific HTTPException
        # from here, but for now, we'll return an empty list.
        return []

# ==============================================================================
# EFO Semantic Search Logic
# ==============================================================================

# ---------- Load EFO data at startup -----------------
EFO_EMBEDDINGS_PATH = 'data/efo_embeddings.parquet'
EFO_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

logger.info("Loading EFO embeddings for semantic search...")
try:
    efo_df = pd.read_parquet(EFO_EMBEDDINGS_PATH)
    efo_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    efo_model = SentenceTransformer(EFO_MODEL_NAME, device=efo_device)
    efo_embeddings = torch.tensor(np.array(efo_df['embedding'].tolist()), device=efo_device, dtype=torch.float32)
    logger.info(f"Loaded {len(efo_df)} EFO embeddings on device '{efo_device}'.")
except FileNotFoundError:
    logger.error(f"EFO embeddings file not found at '{EFO_EMBEDDINGS_PATH}'. EFO search will not work.")
    efo_df = pd.DataFrame()
    efo_model = None
    efo_embeddings = None


def semantic_search_efo(query: str, top_k: int = 5):
    """
    Performs a semantic search for a query against the pre-computed EFO embeddings.
    """
    if efo_model is None or efo_df.empty:
        raise RuntimeError("EFO search is not available. Embeddings could not be loaded.")

    # 1. Encode the query to get its embedding
    query_embedding = efo_model.encode(query, convert_to_tensor=True)
    
    # Handle top_k logic
    if top_k < 0:
        k = len(efo_df)
    else:
        k = min(top_k, len(efo_df))

    # 2. Use a utility function to find the top_k most similar embeddings
    hits = util.semantic_search(query_embedding, efo_embeddings, top_k=k)
    
    # The result is a list of lists, one for each query. We only have one.
    hits = hits[0]
    
    results = []
    for hit in hits:
        row_index = hit['corpus_id']
        results.append({
            "efo_id": efo_df.iloc[row_index]['term_id'],
            "label": efo_df.iloc[row_index]['label'],
            "score": hit['score']
        })
        
    return results
