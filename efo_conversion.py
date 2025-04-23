import requests
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_mesh_id_from_file(disease_name, file_path="data/mesh_terms.csv"):
    """
    Retrieves the MeSH DescriptorUI for a given disease name (PreferredTerm) 
    from a CSV file with columns: DescriptorUI, PreferredTerm, SearchTerms.

    Parameters:
        disease_name (str): The disease name (PreferredTerm) to look up.
        file_path (str): Path to the MeSH mapping CSV file.

    Returns:
        str: The corresponding MeSH DescriptorUI.

    Raises:
        ValueError: If the disease name is not found in the file.
    """
    try:
        mesh_df = pd.read_csv(file_path)
        match = mesh_df[mesh_df["PreferredTerm"].str.lower() == disease_name.strip().lower()]
        if match.empty:
            raise ValueError(f"No MeSH ID found for disease name: {disease_name}")
        return match.iloc[0]["DescriptorUI"]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading MeSH file: {e}")

import requests

def get_oxo_mapping_from_mesh(mesh_id):
    """
    Maps a MeSH ID to an EFO ID using the EMBL-EBI Ontology Xref Service (OxO).
    If no EFO mapping is found at distance 1–3, it then tries MONDO.

    Parameters:
        mesh_id (str): The MeSH ID to be mapped.

    Returns:
        str: The corresponding CURIE (e.g., "EFO:0005148" or "MONDO:0001234").

    Raises:
        ValueError: If no mapping is found within distance 1–3 for either ontology.
    """
    oxo_url = 'https://www.ebi.ac.uk/spot/oxo/api/search'
    headers = {'Content-Type': 'application/json'}

    # Try first for EFO, then for MONDO
    for target in ["EFO", "MONDO"]:
        for distance in range(1, 4):
            payload = {
                "ids": [mesh_id],
                "inputSource": "MeSH",
                "mappingTarget": [target],
                "distance": distance
            }
            resp = requests.post(oxo_url, headers=headers, json=payload)
            resp.raise_for_status()
            results = resp.json().get('_embedded', {}).get('searchResults', [])
            if results:
                mappings = results[0].get('mappingResponseList', [])
                if mappings:
                    return mappings[0]['curie']

    raise ValueError(f"No EFO or MONDO mapping found for MeSH ID {mesh_id} within distance 1–3")