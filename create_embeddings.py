import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

def generate_embeddings_from_sssom(file_path: str, model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO'):
    """
    Reads an SSOM TSV file, extracts object IDs and labels, and generates text embeddings.

    Args:
        file_path (str): The path to the SSSOM TSV file.
        model_name (str): The name of the sentence-transformer model to use from Hugging Face.

    Returns:
        pandas.DataFrame: A DataFrame containing 'object_id', 'object_label', 
                          and the generated 'embedding'. Returns an empty DataFrame on error.
    """
    print(f"Loading data from: {file_path}")
    
    try:
        # SSSOM files often contain metadata at the beginning, commented out with '#'.
        # We can use pandas to automatically skip these commented lines.
        df = pd.read_csv(file_path, sep='\t', comment='#')
        
        # 1. Extract the required columns
        if 'object_id' not in df.columns or 'object_label' not in df.columns:
            print("Error: 'object_id' or 'object_label' columns not found in the file.")
            return pd.DataFrame()
            
        object_df = df[['object_id', 'object_label']].copy()
        print(f"Successfully extracted {len(object_df)} total rows.")

        # Drop rows where the label is missing, as they cannot be embedded.
        object_df.dropna(subset=['object_label'], inplace=True)
        print(f"Processing {len(object_df)} rows after removing entries with no object label.")
        
        if object_df.empty:
            print("No data left to process after cleaning. Exiting.")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return pd.DataFrame()

    # 2. Use a Hugging Face model to create text embeddings
    print(f"\nLoading sentence transformer model: '{model_name}'...")
    # This will download the model from the Hugging Face Hub on its first run.
    model = SentenceTransformer(model_name)
    
    # Check if a GPU is available and use it, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: '{device}'")

    # Prepare the list of labels for embedding
    labels_to_embed = object_df['object_label'].tolist()

    print(f"\nGenerating embeddings for {len(labels_to_embed)} labels... (This may take a moment)")
    
    # The encode method computes the embeddings. 
    # show_progress_bar=True provides a visual indicator of progress.
    embeddings = model.encode(
        labels_to_embed, 
        show_progress_bar=True,
        convert_to_tensor=True # More efficient to work with tensors
    )
    
    print("Embedding generation complete.")

    # Add the embeddings back to the DataFrame
    # We convert tensors to lists of floats for easier storage/viewing in pandas.
    object_df['embedding'] = [emb.cpu().numpy().tolist() for emb in embeddings]
    
    return object_df

if __name__ == '__main__':
    # Before running, make sure you have the required libraries installed:
    # pip install pandas
    # pip install sentence-transformers
    # pip install torch
    
    # --- Configuration ---
    # Make sure this file is in the same directory as the script, or provide the full path.
    SSOM_FILE_PATH = '/Users/sumanth/Downloads/mappings_sssom/efo.ols.sssom.tsv' 
    
    # --- Run the process ---
    embeddings_df = generate_embeddings_from_sssom(SSOM_FILE_PATH)
    
    if not embeddings_df.empty:
        print("\n--- Embedding Generation Successful ---")
        print("Here are the first 5 rows of the resulting DataFrame:")
        print(embeddings_df.head())
        
        # You can inspect a single embedding like this:
        print("\n--- Example of a single embedding ---")
        print(f"Label: '{embeddings_df.iloc[0]['object_label']}'")
        # We'll print the first 10 values of the vector for brevity
        embedding_sample = embeddings_df.iloc[0]['embedding'][:10]
        print(f"First 10 values of its embedding vector: {embedding_sample}")
        print(f"Total vector dimension: {len(embeddings_df.iloc[0]['embedding'])}")

        # To save the results to a new file (e.g., CSV or Pickle for preserving list structure):
        # embeddings_df.to_csv('object_embeddings.csv', index=False)
        # embeddings_df.to_pickle('object_embeddings.pkl')
        # print("\nDataFrame saved to 'object_embeddings.pkl'")

