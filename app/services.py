import re, json, logging, pathlib
from typing import List, Set
from itertools import chain
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from safetensors.numpy import load_file

logger = logging.getLogger(__name__)

# load HGNC cache as before …
HGNC_CACHE: Set[str] = set()
try:
    cache_path = pathlib.Path(__file__).parent.parent / "data" / "gene_names.txt"
    if cache_path.exists():
        HGNC_CACHE = {ln.strip() for ln in cache_path.read_text().splitlines() if ln.strip()}
except Exception as e:
    logger.warning("Could not load HGNC cache: %s", e)

# your regex
_gene_regex = re.compile(r"\b[A-Z][A-Z0-9\-]{1,9}\b")

def extract_gene_names(text: str) -> List[str]:
    return list(dict.fromkeys(_gene_regex.findall(text)))

def extract_genes_with_chatgpt(
    text: str,
    context_label: str,
    llm,
    extra_hgnc_check: bool = True,
    max_chunk_chars: int = 15_000
) -> List[str]:
    """
    1) If text ≤ max_chunk_chars, send in one shot.
    2) Else split on sentences into ~max_chunk_chars chunks.
    3) Call LLM (GPT‑4.1) on each, JSON‑parse, uppercase & dedupe.
    4) Finally apply regex‐in‐text + optional HGNC cache.
    """
    prompt = f"""
You are an expert molecular biologist.
Extract *only* true human gene symbols from the following {context_label}. Follow these rules exactly:
1.  A valid symbol must match the regex  `\\b[A-Z][A-Z0-9\\-]{1,9}\\b`.
2.  The 3 words **before** or **after** the token must mention biology
    (e.g. “gene”, “expression”, “mutation”, “variant”, “allele”, “up‑regulated”).
3.  Exclude:
      • cell‑lines (e.g. HEK293, 3T3)
      • protein names that are not also gene symbols
      • pathways, GO terms, locus tags, accession IDs
      • gene names that appear ONLY in the bibliography or references and not the main text 
            (e.g. in a citation such as: "Kettunen E, ... L1CAM, INP10, P-cadherin, tPA and ITGB4 over-expression in malignant pleural mesothe- liomas revealed by combined use of cDNA and
            tissue microarray. Carcinogenesis 2005; 26: 17-25." If the genes such as L1CAM INP10 DO NOT appear in the main text, exclude them from the list)
4.  Cross‑check against the official HGNC list (human genes) and discard
    anything absent.
5.  **Output JSON array** of the unique symbols, lowercase *not* allowed.
   e.g.  ["BRCA1","TP53"]

Respond with **nothing but** that JSON.
"""

    # helper to split on sentence boundaries
    def chunk_text(s: str):
        if len(s) <= max_chunk_chars:
            yield s
            return
        parts = re.split(r'(?<=[\.\?\!])\s+', s)
        buf = ""
        for sent in parts:
            if len(buf) + len(sent) + 1 <= max_chunk_chars:
                buf += sent + " "
            else:
                yield buf
                buf = sent + " "
        if buf:
            yield buf

    raw_lists = []
    for chunk in chunk_text(text[:50_000]):  # cap at 50k chars
        messages = [
            {"role": "system", "content": "You are a precise extraction engine."},
            {"role": "user",   "content": prompt},
            {"role": "user",   "content": chunk}
        ]
        try:
            raw = llm.predict_messages(messages).content.strip()
            arr = json.loads(raw)
            if isinstance(arr, list):
                raw_lists.append([g.strip().upper() for g in arr])
        except Exception:
            logger.warning("Chunk output not valid JSON, skipping.")

    # flatten & preserve order
    model_genes = list(dict.fromkeys(chain.from_iterable(raw_lists)))

    # must actually appear in text
    in_text = set(extract_gene_names(text))
    filtered = [g for g in model_genes if g in in_text]

    # optional HGNC check
    if extra_hgnc_check and HGNC_CACHE:
        filtered = [g for g in filtered if g in HGNC_CACHE]

    return filtered

# ---------- Autocomplete Logic ----------
DATA_DIR = "data"
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

def autocomplete(q: str, top_n: int = 8):
    if not q.strip():
        return []

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