import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from safetensors.numpy import save_file, load_file
os.environ['CURL_CA_BUNDLE'] = ''

CSV_FILE = "data/mesh_terms.csv"
PREFERRED_EMBEDDINGS_FILE = "data/preferred_embeddings.safetensors"
ALT_EMBEDDINGS_FILE = "data/alt_embeddings.safetensors"
TERMS_FILE = "data/mesh_terms.json"

@st.cache_resource
def load_semantic_model():
    model = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    return model

def precompute_embeddings(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    preferred_terms = df["PreferredTerm"].tolist()
    alt_terms = df["SearchTerms"].tolist()
    model = load_semantic_model()
    if os.path.exists(PREFERRED_EMBEDDINGS_FILE) and os.path.exists(ALT_EMBEDDINGS_FILE) and os.path.exists(TERMS_FILE):
        preferred_embeddings = load_file(PREFERRED_EMBEDDINGS_FILE)["embeddings"]
        alt_embeddings = load_file(ALT_EMBEDDINGS_FILE)["embeddings"]
        with open(TERMS_FILE, "r", encoding="utf-8") as f:
            terms_dict = json.load(f)
        preferred_terms = terms_dict.get("preferred_terms", preferred_terms)
        alt_terms = terms_dict.get("alt_terms", alt_terms)
    else:
        st.info("Precomputing embeddings... (this may take a moment)")
        preferred_embeddings = model.encode(preferred_terms, convert_to_numpy=True, show_progress_bar=True)
        alt_embeddings = model.encode(alt_terms, convert_to_numpy=True, show_progress_bar=True)
        save_file({"embeddings": preferred_embeddings}, PREFERRED_EMBEDDINGS_FILE)
        save_file({"embeddings": alt_embeddings}, ALT_EMBEDDINGS_FILE)
        terms_dict = {"preferred_terms": preferred_terms, "alt_terms": alt_terms}
        with open(TERMS_FILE, "w", encoding="utf-8") as f:
            json.dump(terms_dict, f)
    return preferred_terms, alt_terms, preferred_embeddings, alt_embeddings

@st.cache_data
def load_mesh_data():
    return precompute_embeddings()

@st.cache_data
def load_mesh_mapping(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    mapping = dict(zip(df["PreferredTerm"], df["DescriptorUI"]))
    return mapping

def vectorized_cosine_similarity(query_emb, embeddings):
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return np.dot(norm_embeddings, query_norm)

def search_mesh(query: str, preferred_terms, alt_terms, preferred_embeddings, alt_embeddings, 
                top_n: int = 5, 
                weight_semantic: float = 0.7, weight_fuzzy: float = 0.3,
                fuzzy_weight_pref: float = 0.5, fuzzy_weight_alt: float = 0.5):
    if not query:
        return []
    model = load_semantic_model()
    query_embedding = model.encode(query, convert_to_numpy=True)
    semantic_sims = vectorized_cosine_similarity(query_embedding, preferred_embeddings)
    fuzzy_sims_pref = np.array([fuzz.partial_ratio(query.lower(), term.lower()) / 100.0 
                                 for term in preferred_terms])
    fuzzy_sims_alt = np.array([fuzz.partial_ratio(query.lower(), term.lower()) / 100.0 
                                for term in alt_terms])
    combined_fuzzy = fuzzy_weight_pref * fuzzy_sims_pref + fuzzy_weight_alt * fuzzy_sims_alt
    combined_scores = weight_semantic * semantic_sims + weight_fuzzy * combined_fuzzy
    top_indices = np.argsort(combined_scores)[::-1][:top_n]
    return [preferred_terms[i] for i in top_indices]
