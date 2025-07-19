import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os

# --- Configuration ---
EFO_EMBEDDINGS_PATH = 'data/subject_embeddings.parquet'
GSE_METADATA_PATH = 'data/human-gse-processed-meta.json'
FINAL_LOOKUP_PATH = 'data/efo_to_gse_lookup.json' # The final, important output
MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

def generate_and_save_associations():
    """
    Main function to perform the one-time processing and save the final lookup file.
    """
    # Ensure the output directory exists
    os.makedirs('data', exist_ok=True)
    
    # --- Step 1: Load Model ---
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded on device: '{device}'")

    # --- Step 2: Load and Process EFO Data ---
    print("\n--- Loading EFO Data ---")
    try:
        efo_df = pd.read_parquet(EFO_EMBEDDINGS_PATH)
        # Explicitly set dtype for MPS compatibility
        efo_embeddings = torch.tensor(np.array(efo_df['embedding'].tolist()), device=device, dtype=torch.float32)
        print(f"Loaded {len(efo_df)} pre-computed EFO embeddings.")
    except FileNotFoundError:
        print(f"Error: EFO embeddings file not found at '{EFO_EMBEDDINGS_PATH}'.")
        print("Please generate this file first using your previous script.")
        return

    # --- Step 3: Load and Process GSE Data ---
    print("\n--- Loading and Processing GSE Data ---")
    with open(GSE_METADATA_PATH, 'r') as f:
        gse_data = json.load(f)

    records = []
    for gse_id, details in gse_data.items():
        # Use the first title as the representative title for the study
        if details.get('titles'):
            first_title = next(iter(details['titles'].values()))
            records.append({
                'gse_id': gse_id,
                'title': first_title,
                'silhouette_score': details.get('silhouette_score', -1.0)
            })
    gse_df = pd.DataFrame(records)
    print(f"Loaded {len(gse_df)} GSE studies.")

    print("Generating embeddings for GSE titles...")
    gse_embeddings = model.encode(gse_df['title'].tolist(), show_progress_bar=True, convert_to_tensor=True)

    # --- Step 4: Rank and Create Final Lookup File ---
    print("\n--- Finding and Ranking Associations ---")
    
    # Normalize silhouette scores (0 to 1 range)
    min_s = gse_df['silhouette_score'].min()
    max_s = gse_df['silhouette_score'].max()
    if (max_s - min_s) == 0:
        gse_df['norm_silhouette'] = 0.5
    else:
        gse_df['norm_silhouette'] = (gse_df['silhouette_score'] - min_s) / (max_s - min_s)

    # Perform semantic search
    all_hits = util.semantic_search(efo_embeddings, gse_embeddings, top_k=100) # Get more results to re-rank

    final_lookup = {}
    for i, efo_row in efo_df.iterrows():
        efo_id = efo_row['subject_id']
        hits = all_hits[i]
        ranked_results = []
        for hit in hits:
            gse_index = hit['corpus_id']
            semantic_score = hit['score']
            silhouette_score = gse_df.iloc[gse_index]['norm_silhouette']
            
            # Weighted final score
            final_score = (0.7 * semantic_score) + (0.3 * silhouette_score)
            
            ranked_results.append({
                'gse_id': gse_df.iloc[gse_index]['gse_id'],
                'score': final_score
            })
        
        # Sort by the combined score and keep the top 20
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        final_lookup[efo_id] = ranked_results[:20]

    # --- Step 5: Save the Final Output ---
    print(f"\n--- Saving Final Lookup File ---")
    with open(FINAL_LOOKUP_PATH, 'w') as f:
        json.dump(final_lookup, f, indent=4)
    print(f"Process complete. Lookup file saved to '{FINAL_LOOKUP_PATH}'")

    # --- Display a sample of the results ---
    print("\n--- Example of Saved Data Structure ---")
    example_efo_id = next(iter(final_lookup))
    print(f"'{example_efo_id}': {json.dumps(final_lookup[example_efo_id][:2], indent=4)}")


if __name__ == '__main__':
    generate_and_save_associations()
