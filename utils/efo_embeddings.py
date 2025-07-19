import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def generate_embeddings_from_sssom(file_path: str, model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO'):
    """
    Reads an SSOM TSV file, extracts subject IDs and labels, and generates text embeddings.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#')
        subject_df = df[['subject_id', 'subject_label']].copy()
        subject_df.dropna(subset=['subject_label'], inplace=True)
        subject_df.drop_duplicates(subset=['subject_id', 'subject_label'], keep='first', inplace=True)
        print(f"Successfully extracted and cleaned {len(subject_df)} rows.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return pd.DataFrame(), None

    print(f"\nLoading sentence transformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: '{device}'")
    
    labels_to_embed = subject_df['subject_label'].tolist()
    print(f"\nGenerating embeddings for {len(labels_to_embed)} labels...")
    embeddings = model.encode(
        labels_to_embed, 
        show_progress_bar=True,
        convert_to_tensor=True
    )
    subject_df['embedding'] = embeddings.cpu().tolist() # Store as list
    
    # Return the model and embeddings tensor for the search step
    return subject_df, model, embeddings

def search(query: str, model, embeddings_df, corpus_embeddings, top_k=5):
    """
    Performs a semantic search for a query against a corpus of embeddings.
    """
    print(f"\n🔎 Searching for: '{query}'")
    
    # 1. Encode the query to get its embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 2. Use a utility function to find the top_k most similar embeddings
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    
    # The result is a list of lists, one for each query. We only have one.
    hits = hits[0] 
    
    print(f"Top {top_k} results:")
    for hit in hits:
        # Get the row from the original DataFrame
        row_index = hit['corpus_id']
        label = embeddings_df.iloc[row_index]['subject_label']
        score = hit['score']
        subject_id = embeddings_df.iloc[row_index]['subject_id']
        print(f"  - Score: {score:.4f}\t Label: {label} (ID: {subject_id})")

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    FILE_PATH = 'data/efo.ols.sssom.tsv'  # Corrected relative path
    
    # 1. Generate the embeddings
    embeddings_df, model, corpus_embeddings = generate_embeddings_from_sssom(FILE_PATH)
    
    if not embeddings_df.empty:
        print("\n--- Embedding Generation Successful ---")
        print(embeddings_df.head())
        
        # 2. Perform semantic search tests
        print("\n--- Testing Semantic Search ---")
        search(query="cancer of the lung", model=model, embeddings_df=embeddings_df, corpus_embeddings=corpus_embeddings, top_k=5)
        search(query="disease of the heart muscle", model=model, embeddings_df=embeddings_df, corpus_embeddings=corpus_embeddings, top_k=5)

        # To save the results to a new file:
        print("\nSaving DataFrame to 'data/subject_embeddings.parquet'")
        embeddings_df.to_parquet('data/subject_embeddings.parquet', index=False)
        print("Save complete.")