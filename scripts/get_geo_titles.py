import requests
import xml.etree.ElementTree as ET
import time
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -- Configuration --
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
SEARCH_TERM = '"Homo sapiens"[Organism] AND (gse[ETYP] OR gds[ETYP])'

# --- DYNAMIC PATH CONFIGURATION ---
# This logic ensures the output path is correct whether the script is run
# from 'getgpt/' or 'getgpt/setup/'.

# Get the absolute path of the directory where the script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if the script is in a 'setup' subdirectory.
if os.path.basename(script_dir) == 'setup':
    # If yes, the project root is one level up.
    project_root = os.path.dirname(script_dir)
else:
    # Otherwise, the script is already in the project root.
    project_root = script_dir

# Construct the final, absolute path for the output file.
OUTPUT_FILE = os.path.join(project_root, 'data', 'human_study_titles.tsv')


# --- Parallel Processing Config ---
BATCH_SIZE = 200     # How many IDs to fetch in a single request
MAX_WORKERS = 8      # Number of parallel threads in the pool
MAX_RETRIES = 5      # Max retries for a failed batch

# --- Rate Limiting Config ---
# Allows only 1 request at a time to reliably stay under NCBI's limit
rate_limiter = threading.Semaphore(1)

def get_record_ids():
    """Searches GEO to get all specified record IDs and returns them."""
    print("🔬 Searching for human studies...")
    search_params = {"db": "gds", "term": SEARCH_TERM, "usehistory": "y"}
    response = requests.post(ESEARCH_URL, data=search_params)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    count = int(root.find("Count").text)
    webenv = root.find("WebEnv").text
    query_key = root.find("QueryKey").text
    print(f"✅ Found {count:,} records. Fetching all UIDs...")
    fetch_params = {"db": "gds", "query_key": query_key, "WebEnv": webenv, "retstart": 0, "retmax": count}
    response = requests.post(ESEARCH_URL, data=fetch_params)
    response.raise_for_status()
    id_root = ET.fromstring(response.content)
    id_list = [elem.text for elem in id_root.findall(".//Id")]
    return id_list

def fetch_one_batch(id_string):
    """Worker function to fetch and parse a single batch with rate limiting and retries."""
    with rate_limiter: # This ensures only one thread can proceed at a time
        for attempt in range(MAX_RETRIES):
            # Pause to stay under the 3 requests/sec limit
            time.sleep(0.4)
            try:
                summary_params = {"db": "gds", "id": id_string}
                response = requests.post(ESUMMARY_URL, data=summary_params)
                response.raise_for_status()
                
                summaries = []
                root = ET.fromstring(response.content)
                # This corrected parsing logic is the key to making this work
                for doc_sum in root.findall(".//DocSum"):
                    accession_item = doc_sum.find("Item[@Name='GSE']")
                    if accession_item is None:
                        accession_item = doc_sum.find("Item[@Name='GDS']")
                    title_item = doc_sum.find("Item[@Name='title']")
                    if accession_item is not None and title_item is not None:
                        summaries.append({
                            "Accession": accession_item.text,
                            "Title": title_item.text
                        })
                return summaries # Success
            
            except requests.exceptions.HTTPError as e:
                # Handle "Too Many Requests" error with exponential backoff
                if e.response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Got 429 error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else: # For other HTTP errors, fail the batch
                    print(f"HTTP error on batch: {e}")
                    break # Exit retry loop
        
    print(f"Batch failed after {MAX_RETRIES} retries.")
    return []

def fetch_summaries_in_parallel(id_list):
    """Orchestrates fetching all summaries in parallel using a thread pool."""
    all_summaries = []
    # Split the list of all UIDs into smaller batches
    batches = [",".join(id_list[i:i+BATCH_SIZE]) for i in range(0, len(id_list), BATCH_SIZE)]
    print(f"🚚 Fetching {len(id_list):,} summaries in {len(batches)} batches...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each batch
        future_to_batch = {executor.submit(fetch_one_batch, batch): batch for batch in batches}
        
        # Process futures as they complete, with a progress bar
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Fetching Batches"):
            try:
                batch_summaries = future.result()
                all_summaries.extend(batch_summaries)
            except Exception as e:
                print(f"A batch failed with an unexpected error: {e}")

    return all_summaries

def write_to_tsv(summaries):
    """Writes the fetched data to a TSV file."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Accession", "Title"], delimiter='\t')
        writer.writeheader()
        writer.writerows(summaries)
    
    # The OUTPUT_FILE variable is now an absolute path, so this is always correct.
    print(f"\n🎉 Done! Data for {len(summaries):,} records saved to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    try:
        record_ids = get_record_ids()
        if record_ids:
            summary_data = fetch_summaries_in_parallel(record_ids)
            if summary_data:
                write_to_tsv(summary_data)
            else:
                print("\n❌ Fetching completed, but no summary data was retrieved. The output file will not be created.")
        else:
            print("No records found matching the query.")
    except requests.exceptions.RequestException as e:
        print(f"A critical error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")