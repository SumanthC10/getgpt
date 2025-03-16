import time
import numpy as np
import requests
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

def ds_get_disgenet_geneset_to_disease(gene_symbols, disease_ids, DISGENET_API_KEY, pageNumber=0):
    payload = {
        "commonGenes": len(gene_symbols),
        "disList": ",".join(disease_ids),
        "geneHGNCList": ",".join(gene_symbols),
        "maxPvalue": 1,
        "minScore": 0,
        "maxScore": 1,
        "pageNumber": pageNumber,
        "source": "ALL"
    }
    headers = {
        "Authorization": DISGENET_API_KEY,
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    url = "https://api.disgenet.com/api/v1/enrichment/gene"
    response = requests.post(url, json=payload, headers=headers, verify=False)
    
    # Handle rate limiting
    while response.status_code == 429:
        wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 60))
        st.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        response = requests.post(url, json=payload, headers=headers, verify=False)
        
    if response.ok:
        return response.json()
    else:
        st.error(f"Error in geneset to disease query: {response.text}")
        return None

def ds_load_semantic_model():
    # Load the semantic model only once
    model = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    return model

def ds_select_best_umls_term(mesh_term: str, umls_terms: list, weight_semantic: float = 0.7, weight_fuzzy: float = 0.3):
    """
    Given an input MeSH term and a list of UMLS terms returned from DisGeNET,
    select the best matching UMLS term based on a weighted combination of semantic similarity
    (cosine similarity) and fuzzy matching.
    """
    if not mesh_term or not umls_terms:
        return None

    model = ds_load_semantic_model()
    # Compute embedding for the original MeSH term
    mesh_embedding = model.encode(mesh_term, convert_to_numpy=True)
    # Compute embeddings for each UMLS term
    umls_embeddings = model.encode(umls_terms, convert_to_numpy=True)

    # Compute cosine similarities
    mesh_norm = mesh_embedding / (np.linalg.norm(mesh_embedding) + 1e-8)
    norms = umls_embeddings / (np.linalg.norm(umls_embeddings, axis=1, keepdims=True) + 1e-8)
    semantic_sims = np.dot(norms, mesh_norm)

    # Compute fuzzy matching scores (scaled to 0-1)
    fuzzy_sims = np.array([fuzz.partial_ratio(mesh_term.lower(), term.lower()) / 100.0 for term in umls_terms])

    # Combine scores using the provided weights
    combined_scores = weight_semantic * semantic_sims + weight_fuzzy * fuzzy_sims
    best_index = np.argmax(combined_scores)
    return umls_terms[best_index]
