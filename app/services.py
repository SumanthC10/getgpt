import re, json, logging, pathlib
from typing import List, Set
from itertools import chain
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from safetensors.numpy import load_file

logger = logging.getLogger(__name__)

# ==============================================================================
# EFO Semantic Search Logic
# ==============================================================================

# ---------- Load EFO data at startup -----------------
EFO_EMBEDDINGS_PATH = 'data/subject_embeddings.parquet'
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
            "efo_id": efo_df.iloc[row_index]['subject_id'],
            "label": efo_df.iloc[row_index]['subject_label'],
            "score": hit['score']
        })
        
    return results
