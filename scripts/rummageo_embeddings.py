import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os

# --- Configuration ---
EFO_EMBEDDINGS_PATH = 'data/efo_embeddings.parquet'
STUDY_TITLES_PATH = 'data/human_study_titles.tsv'
GSE_METADATA_PATH = 'data/human-gse-processed-meta.json'
FINAL_LOOKUP_PATH = 'data/efo_to_gse_lookup.json'
MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

# --- Configuration for Two-Stage Filtering ---
SEMANTIC_TOP_K = 100              # Initial pool of candidates from semantic search
SEMANTIC_SCORE_THRESHOLD = 0.8   # The '0.X' threshold for semantic relevance
FINAL_TOP_N_BY_QUALITY = 10       # Final number of studies to keep after quality ranking

def generate_and_save_associations():
    """
    Main function to perform the one-time processing and save the final lookup file.
    """
    os.makedirs('data', exist_ok=True)
    
    # --- Step 1: Load Model ---
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded on device: '{device}'")

    # --- Step 2: Load and Process EFO Data ---
    print("\n--- Loading EFO Data ---")
    try:
        efo_df = pd.read_parquet(EFO_EMBEDDINGS_PATH)
        efo_embeddings = torch.tensor(np.array(efo_df['embedding'].tolist()), device=device, dtype=torch.float32)
        print(f"Loaded {len(efo_df)} pre-computed EFO embeddings.")
    except FileNotFoundError:
        print(f"Error: EFO embeddings file not found at '{EFO_EMBEDDINGS_PATH}'.")
        return

    # --- Step 3: Filter, Load, and Merge Study Data ---
    print("\n--- Filtering and Loading Study Data ---")
    try:
        gse_df = pd.read_csv(STUDY_TITLES_PATH, sep='\t')
        gse_df.rename(columns={'Accession': 'gse_id', 'Title': 'title'}, inplace=True)
        gse_df['gse_id'] = 'GSE' + gse_df['gse_id'].astype(str)
        
        with open(GSE_METADATA_PATH, 'r') as f:
            gse_meta_data = json.load(f)
            
        tsv_ids = set(gse_df['gse_id'])
        json_ids = set(gse_meta_data.keys())
        common_ids = tsv_ids.intersection(json_ids)
        
        print(f"Found {len(tsv_ids)} studies in TSV, {len(json_ids)} in JSON. Common studies: {len(common_ids)}.")
        
        gse_df = gse_df[gse_df['gse_id'].isin(common_ids)].copy()
        
        silhouette_scores = {gse_id: details.get('silhouette_score', -1.0) 
                             for gse_id, details in gse_meta_data.items()}
                             
        gse_df['silhouette_score'] = gse_df['gse_id'].map(silhouette_scores)
        gse_df.dropna(subset=['title', 'silhouette_score'], inplace=True)
        
        print(f"Processing {len(gse_df)} studies with both a title and silhouette score.")

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found: {e.filename}")
        return

    print("Generating embeddings for study titles...")
    gse_embeddings = model.encode(gse_df['title'].tolist(), show_progress_bar=True, convert_to_tensor=True)

    # --- Step 4: Two-Stage Filtering and Ranking ---
    print("\n--- Applying Two-Stage Filtering and Ranking ---")
    
    all_hits = util.semantic_search(efo_embeddings, gse_embeddings, top_k=SEMANTIC_TOP_K)

    final_lookup = {}
    for i, efo_row in efo_df.iterrows():
        efo_id = efo_row['term_id']
        hits = all_hits[i]
        
        semantic_candidates = []
        for hit in hits:
            if hit['score'] >= SEMANTIC_SCORE_THRESHOLD:
                gse_index = hit['corpus_id']
                semantic_candidates.append({
                    'gse_id': str(gse_df.iloc[gse_index]['gse_id']),
                    'semantic_score': hit['score'],
                    'silhouette_score': gse_df.iloc[gse_index]['silhouette_score']
                })
        
        semantic_candidates.sort(key=lambda x: x['silhouette_score'], reverse=True)
        
        top_by_quality = semantic_candidates[:FINAL_TOP_N_BY_QUALITY]
        
        final_ranked_results = [
            {
                'gse_id': item['gse_id'],
                'semantic_score': round(item['semantic_score'], 4),
                'silhouette_score': round(item['silhouette_score'], 4)
            } 
            for item in top_by_quality
        ]
        
        final_lookup[efo_id] = final_ranked_results

    # --- Step 5: Save the Final Output ---
    print(f"\n--- Saving Final Lookup File ---")
    with open(FINAL_LOOKUP_PATH, 'w') as f:
        json.dump(final_lookup, f, indent=4)
    print(f"Process complete. Lookup file saved to '{FINAL_LOOKUP_PATH}'")

    # --- Display a sample of the results ---
    if final_lookup:
        print("\n--- Example of Saved Data Structure ---")
        example_efo_id = next(iter(final_lookup))
        if final_lookup[example_efo_id]:
             print(f"'{example_efo_id}': {json.dumps(final_lookup[example_efo_id][:2], indent=4)}")
        else:
             print(f"'{example_efo_id}': []  (No studies met the filtering criteria)")
        
        # --- MODIFIED: Calculate and print summary statistics ---
        counts_per_term = [len(studies) for studies in final_lookup.values()]
        all_saved_studies = [study for studies_list in final_lookup.values() for study in studies_list]
        total_efo_terms = len(final_lookup)

        print(f"\n--- Summary Statistics ---")

        if total_efo_terms > 0:
            terms_with_no_studies = sum(1 for studies in final_lookup.values() if not studies)
            print(f"EFO terms with no studies found: {terms_with_no_studies} / {total_efo_terms}")
        
        if counts_per_term:
            mean_studies = np.mean(counts_per_term)
            print(f"Mean studies saved per EFO term: {mean_studies:.2f}")

        if all_saved_studies:
            mean_semantic = np.mean([s['semantic_score'] for s in all_saved_studies])
            mean_silhouette = np.mean([s['silhouette_score'] for s in all_saved_studies])
            print(f"Overall average semantic score: {mean_semantic:.4f}")
            print(f"Overall average silhouette score: {mean_silhouette:.4f}")


if __name__ == '__main__':
    generate_and_save_associations()