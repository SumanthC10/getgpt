import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def generate_embeddings(file_path: str, model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO'):
    """
    Reads a prepared ontology TSV, combines label, definition, and synonyms,
    and generates text embeddings.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#')
        df.rename(columns={'id': 'term_id'}, inplace=True)
        # <<< NEW: Handle missing values in the new synonyms column
        df['definition'] = df['definition'].fillna('')
        df['synonyms'] = df['synonyms'].fillna('')
        df.dropna(subset=['label'], inplace=True)
        df.drop_duplicates(subset=['term_id'], keep='first', inplace=True)
        print(f"Successfully loaded and cleaned {len(df)} rows.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return pd.DataFrame(), None, None

    print("\nCombining labels, definitions, and synonyms for embedding...")
    def combine_text(row):
        # Create a text block with the label, synonyms (if they exist), and the definition
        if row['synonyms']:
            return f"{row['label']} (also known as: {row['synonyms']}): {row['definition']}"
        else:
            return f"{row['label']}: {row['definition']}"

    df['text_to_embed'] = df.apply(combine_text, axis=1)
    
    print(f"\nLoading sentence transformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: '{device}'")
    
    embeddings = model.encode(
        df['text_to_embed'].tolist(), 
        show_progress_bar=True, 
        convert_to_tensor=True
    )
    df['embedding'] = embeddings.cpu().tolist()
    return df, model, embeddings

def search(query: str, model, embeddings_df, corpus_embeddings, top_k=5):
    print(f"\n🔎 Searching for: '{query}'")
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    print(f"Top {top_k} results:")
    if not hits:
        print("  - No results found.")
        return
    for hit in hits:
        row_index = hit['corpus_id']
        label = embeddings_df.iloc[row_index]['label']
        score = hit['score']
        term_id = embeddings_df.iloc[row_index]['term_id']
        print(f"  - Score: {score:.4f}\tID: {term_id}\tLabel: {label}")

if __name__ == '__main__':
    FILE_PATH = 'data/efo_ontology_terms.tsv'
    embeddings_df, model, corpus_embeddings = generate_embeddings(FILE_PATH)
    if model is not None:
        print("\n--- ✅ Embedding Generation Successful ---")
        print("\n--- 🧪 Testing Semantic Search ---")
        search(query="heart attack", model=model, embeddings_df=embeddings_df, corpus_embeddings=corpus_embeddings)
        search(query="cancer of the lung", model=model, embeddings_df=embeddings_df, corpus_embeddings=corpus_embeddings)
        output_path = 'data/efo_embeddings_with_synonyms.parquet'
        print(f"\n💾 Saving DataFrame with embeddings to '{output_path}'")
        embeddings_df.to_parquet(output_path, index=False)
        print("✅ Save complete.")