# autocomplete_api.py
# -----------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np, json, os
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from safetensors.numpy import load_file

# ---------- paths (same as your semantic_search.py) ----------
DATA_DIR = "data"
CSV_FILE = f"{DATA_DIR}/mesh_terms.csv"
PREF_EMB_FILE = f"{DATA_DIR}/preferred_embeddings.safetensors"
ALT_EMB_FILE  = f"{DATA_DIR}/alt_embeddings.safetensors"
TERMS_FILE    = f"{DATA_DIR}/mesh_terms.json"

# ---------- load everything once at startup -----------------
print("Loading MeSH term vectors …")
preferred_emb = load_file(PREF_EMB_FILE)["embeddings"]
alt_emb       = load_file(ALT_EMB_FILE)["embeddings"]
with open(TERMS_FILE, "r", encoding="utf-8") as f:
    terms_json = json.load(f)
preferred_terms = terms_json["preferred_terms"]
alt_terms       = terms_json["alt_terms"]

model = SentenceTransformer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
print("Autocomplete service ready!")

# ---------- utility -----------------------------------------
def _cosine_batch(qv: np.ndarray, mat: np.ndarray):
    qn = qv / (np.linalg.norm(qv) + 1e-8)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return mn @ qn

# ---------- FastAPI -----------------------------------------
app = FastAPI(title="GetGPT Autocomplete API")

class CompletionResp(BaseModel):
    disease: str
    score: float

@app.get("/v1/autocomplete", response_model=list[CompletionResp])
def autocomplete(
    q: str = Query(..., min_length=2, description="User prefix / query string"),
    top_n: int = Query(8, le=20)
):
    if not q.strip():
        raise HTTPException(400, "Empty query")

    qvec = model.encode(q, convert_to_numpy=True)

    # semantic scores
    sem_scores = _cosine_batch(qvec, preferred_emb)

    # fuzzy scores (rapidfuzz ratio scaled 0‑1)
    fz_pref = np.array([fuzz.partial_ratio(q.lower(), t.lower()) / 100
                        for t in preferred_terms])
    fz_alt  = np.array([fuzz.partial_ratio(q.lower(), t.lower()) / 100
                        for t in alt_terms])

    combined = 0.7 * sem_scores + 0.3 * (0.5 * fz_pref + 0.5 * fz_alt)
    idx = np.argsort(combined)[::-1][:top_n]

    return [
        {"disease": preferred_terms[i], "score": float(combined[i])}
        for i in idx
    ]