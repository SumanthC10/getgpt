import os
import time
import tempfile
import pandas as pd
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from abc import ABC, abstractmethod
import collections
import gseapy as gp
from pandasgwas import get_studies_by_efo_id , get_associations_by_study_id

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

############################################
### PAGER Class for API Requests (FIXED) ###
############################################
class PAGER():
    def __init__(self):
        self.base_url = 'https://discovery.informatics.uab.edu/PAGER/index.php'
        # Define headers once to be used in all requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }
        
    def pagRankedGene(self, pag_id):
        """Retrieves RP-ranked genes with RP-scores for a given PAG ID."""
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.5)
        
        api_url = f"{self.base_url}/genesinPAG/viewgenes/{pag_id}"
        try:
            # Make request with browser headers and a timeout
            response = requests.get(api_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'gene' in data and data['gene']:
                df = pd.DataFrame(data['gene'])
                # FIX: Use the correct, case-sensitive column name 'RP_SCORE'
                if 'RP_SCORE' in df.columns:
                    df['RP_SCORE'] = pd.to_numeric(df['RP_SCORE'])
                return df
            else:
                # Return an empty DataFrame with correct column names
                return pd.DataFrame(columns=['GENE_SYM', 'RP_SCORE'])
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data for {pag_id}: {e}")
            return pd.DataFrame(columns=['GENE_SYM', 'RP_SCORE'])


############################################
### Selenium Functions for Web Scraping ###
############################################
def _build_opts(folder):
    """Configures and returns Selenium Chrome options."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-features=DownloadBubble,DownloadBubbleV2")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    prefs = {
        "download.default_directory": folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "profile.managed_default_content_settings.images": 2,
    }
    opts.add_experimental_option("prefs", prefs)
    return opts

def _wait_for_new_file(existing_files, folder, timeout=20):
    """Waits for a new file to appear in a directory."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        diff = set(os.listdir(folder)) - existing_files
        if diff:
            return diff.pop()
        time.sleep(0.5)
    return None

def get_pager_search_results(gene_input: str) -> pd.DataFrame:
    """
    Runs a PAGER advanced search and downloads the results as a CSV.
    """
    download_dir = tempfile.mkdtemp(prefix="pager_dl_")
    driver = webdriver.Chrome(service=Service(), options=_build_opts(download_dir))
    try:
        print("1. Navigating to PAGER website and submitting gene list...")
        driver.get("https://discovery.informatics.uab.edu/PAGER/index.php/search/advanced")
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "gene_list")))
        box = driver.find_element(By.ID, "gene_list")
        box.clear()
        box.send_keys(gene_input)
        driver.find_element(By.ID, "filter_basic").click()
        
        print("2. Downloading results CSV...")
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Download As']"))).click()
        except Exception:
            pass
        
        existing = set(os.listdir(download_dir))
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.buttons-csv"))).click()
        
        file_name = _wait_for_new_file(existing, download_dir, timeout=15)
        if not file_name:
            raise RuntimeError("Advanced-search CSV did not download.")
        
        adv_path = os.path.join(download_dir, file_name)
        print(f"   Successfully downloaded: {file_name}")
    finally:
        driver.quit()
    
    adv_df = pd.read_csv(adv_path)
    if "PAG ID" not in adv_df.columns:
        raise RuntimeError("Advanced CSV missing 'PAG ID' column.")
    return adv_df

class DataSource(ABC):
    def __init__(self, source_name):
        self.source_name = source_name

    @abstractmethod
    def get_dsource(self):
        pass
    @abstractmethod
    def get_genes(self, disease_id):
        pass

class GWASCatalog(DataSource):
    def __init__(self, source_name="GWAS Catalog"):
        super().__init__(source_name)

    def get_dsource(self):
        # Placeholder for getting data from the GWAS Catalog
        return f"Data from {self.source_name}"
    
    def get_genes(self, disease_id: str) -> pd.DataFrame:
        print(f"\n[DEBUG:GWASCatalog] Received disease_id: {disease_id}")
        try:
            # Normalize EFO ID to use colon
            normalized_efo_id = disease_id.replace(':', '_')
            print(f"[DEBUG:GWASCatalog] Normalized EFO ID: {normalized_efo_id}")
            # 1. Find all studies for the given disease ID
            studies_df = get_studies_by_efo_id(normalized_efo_id).studies
            if studies_df.empty:
                print(f"[DEBUG:GWASCatalog] Info: No studies found for disease ID {normalized_efo_id}")
                return pd.DataFrame()
            print(f"[DEBUG:GWASCatalog] Found {len(studies_df)} studies for {normalized_efo_id}")
        except Exception as e:
            print(f"[DEBUG:GWASCatalog] Error fetching studies for disease ID {normalized_efo_id}: {e}")
            return pd.DataFrame()

        all_results = []
        # 2. Process each study to extract gene associations
        for study_id in studies_df['accessionId']:
            try:
                associations = get_associations_by_study_id(study_id).associations
                if associations.empty:
                    continue

                # Aggregate results within the current study
                study_genes = {}
                for _, row in associations.iterrows():
                    pvalue = row.get('pvalue')
                    loci = row.get('loci')
                    
                    if not isinstance(loci, list) or not loci:
                        continue
                    
                    # Extract rsID and gene name(s) from nested structure
                    rsid = loci[0].get('strongestRiskAlleles', [{}])[0].get('riskAlleleName', '').split('-')[0]
                    reported_genes = loci[0].get('authorReportedGenes', [])
                    
                    if not rsid or not reported_genes:
                        continue
                        
                    evidence_item = {'rsid': rsid, 'study': study_id}

                    for gene_info in reported_genes:
                        gene_symbol = gene_info.get('geneName')
                        if not gene_symbol or gene_symbol.lower() == 'intergenic':
                            continue
                        
                        # Add or update gene entry for this study
                        if gene_symbol not in study_genes:
                            study_genes[gene_symbol] = {'score': pvalue, 'evidence': [evidence_item]}
                        else:
                            study_genes[gene_symbol]['evidence'].append(evidence_item)
                            if pvalue < study_genes[gene_symbol]['score']:
                                study_genes[gene_symbol]['score'] = pvalue
                
                for gene, data in study_genes.items():
                    all_results.append({
                        "gene_symbol": gene,
                        "source": self.source_name,
                        "score": data['score'],
                        "evidence": data['evidence']
                    })

            except Exception as e:
                print(f"Warning: Could not process study {study_id}. Error: {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        # 3. Aggregate results across all studies
        final_df = pd.DataFrame(all_results)
        
        # Define aggregation logic for grouping by gene symbol
        agg_logic = {
            'source': 'first',
            'score': 'min',  # Keep the best (lowest) p-value across all studies
            'evidence': lambda x: sum(x, []) # Concatenate lists of evidence dicts
        }

        aggregated_df = final_df.groupby('gene_symbol').agg(agg_logic).reset_index()

        return aggregated_df



class RummaGEO:
    def __init__(self, source_name="RummaGEO", efo_mapping_file="data/efo.ols.sssom.tsv"):
        self.source_name = source_name
        self.efo_mapping_file = efo_mapping_file

    def _get_disease_name_from_efo_id(self, efo_id: str) -> str | None:
        """Finds the disease name for a given EFO ID from the mapping file."""
        print(f"\n[DEBUG:RummaGEO] Looking up EFO ID: {efo_id}")
        normalized_efo_id = efo_id.replace('_', ':')
        print(f"[DEBUG:RummaGEO] Normalized EFO ID for lookup: {normalized_efo_id}")
        try:
            with open(self.efo_mapping_file, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > 4 and parts[0] == normalized_efo_id:
                        disease_name = parts[4]
                        print(f"[DEBUG:RummaGEO] Found mapping: {normalized_efo_id} -> {disease_name}")
                        return disease_name
            print(f"[DEBUG:RummaGEO] EFO ID {normalized_efo_id} not found in mapping file.")
            return None
        except FileNotFoundError:
            print(f"[DEBUG:RummaGEO] Error: EFO mapping file not found at {self.efo_mapping_file}")
            return None

    def get_dsource(self):
        # Placeholder for fetching data from the RummaGEO
        return f"Data from {self.source_name}"

    def _filter_gsea_with_llm(self, gsea_results: pd.DataFrame, disease_name: str) -> pd.DataFrame:
        """
        Uses an embedding model to select the most relevant GSEA results.
        """
        print("\nINFO: Using embedding model to filter GSEA results.")
        if gsea_results.empty:
            return pd.DataFrame()

        model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        
        # Embed the disease name and the GSEA terms
        disease_embedding = model.encode(disease_name)
        term_embeddings = model.encode(gsea_results['Term'].tolist())
        
        # Calculate cosine similarity
        similarities = cos_sim(disease_embedding, term_embeddings)
        gsea_results['similarity'] = similarities[0]
        
        # Sort by similarity and return the top 20
        return gsea_results.sort_values(by='similarity', ascending=False).head(20)

    def run_gsea_enrichment(self, gene_list: list, disease_name: str) -> pd.DataFrame:
        """
        Performs gene set enrichment analysis using GSEApy against the DisGeNET
        database to find associations between a gene list and a disease.

        Args:
            gene_list: A list of gene symbols.
            disease_name: The disease term to search for in the results.

        Returns:
            A pandas DataFrame containing the complete enrichment results.
            Returns an empty DataFrame if an error occurs.
        """
        if not gene_list:
            print("Error: The provided gene list is empty.")
            return pd.DataFrame()
            
        print(f"\nRunning GSEApy enrichment for '{disease_name}' with {len(gene_list)} genes...")
        
        try:
            # Perform enrichment analysis using the DisGeNET library
            enr_results = gp.enrichr(
                gene_list=gene_list,
                gene_sets='DisGeNET',
                organism='human',
                outdir=None,  # Suppress file output
                cutoff=1      # Return all results regardless of p-value
            )

            if enr_results is None or enr_results.results.empty:
                print("GSEApy analysis returned no results.")
                return pd.DataFrame()

            results_df = enr_results.results
            print("--- GSEApy Analysis Complete ---")
            
            # Search for the specific disease term in the results for logging purposes
            disease_specific_results = results_df[results_df['Term'].str.contains(disease_name, case=False)]

            if not disease_specific_results.empty:
                print(f"Found initial association for '{disease_name}'.")
            else:
                print(f"Could not find a specific string match for '{disease_name}' in GSEA results.")

            return results_df

        except Exception as e:
            print(f"An error occurred during GSEApy analysis: {e}")
            return pd.DataFrame()

    def get_genes(self, disease_id: str) -> pd.DataFrame:
        """
        Finds relevant genes and calculates an evidence-based score using a multi-phase process:
        1. Fetches candidate gene sets related to a disease from RummaGEO.
        2. Filters for the top 20 human studies by silhouette score.
        3. Aggregates genes from these top studies to create a master gene list.
        4. Performs Gene Set Enrichment Analysis (GSEA) on this list.
        5. Filters GSEA results to find the top 20 most relevant terms using an embedding model.
        6. Calculates a final score for each gene based on its association with these top terms.
        """
        print(f"\n[DEBUG:RummaGEO] Received disease_id: {disease_id}")
        disease_name = self._get_disease_name_from_efo_id(disease_id)
        if not disease_name:
            print(f"[DEBUG:RummaGEO] Could not find disease name for EFO ID {disease_id}. Aborting.")
            return pd.DataFrame()

        print(f"[DEBUG:RummaGEO] PHASE 1: Searching for gene sets related to '{disease_name}' ({disease_id}) via RummaGEO...")
        url = "https://rummageo.com/graphql"
        headers = {"Content-Type": "application/json"}

        discovery_query = """
        query GeneSetTermSearchQuery($terms: [String]!) {
          geneSetTermSearch(terms: $terms) {
            nodes {
              silhouetteScore
              geneSetById { id, nodeId, term, species }
            }
          }
        }
        """
        
        try:
            response = requests.post(url, json={
                "operationName": "GeneSetTermSearchQuery",
                "query": discovery_query,
                "variables": {"terms": [disease_name]}
            })
            response.raise_for_status()
            initial_data = response.json()
            
            all_nodes = initial_data.get("data", {}).get("geneSetTermSearch", {}).get("nodes", [])
            if not all_nodes:
                print(f"[DEBUG:RummaGEO] No gene sets found for '{disease_name}' on RummaGEO.")
                return pd.DataFrame()
            print(f"[DEBUG:RummaGEO] Found {len(all_nodes)} initial gene sets from RummaGEO.")

            # --- Filter for Top 20 Human Studies ---
            human_gene_sets = [
                node for node in all_nodes 
                if node.get('geneSetById') and node['geneSetById'].get('species') == 'human'
            ]
            if not human_gene_sets:
                print(f"No human-specific gene sets found for '{disease_name}'.")
                return pd.DataFrame()

            sorted_sets = sorted(human_gene_sets, key=lambda x: x.get('silhouetteScore') or -1, reverse=True)
            top_20_sets = sorted_sets[:20]
            
            # --- PHASE 2: Fetch and Aggregate Genes ---
            print(f"PHASE 2: Fetching genes for the top {len(top_20_sets)} studies...")
            gene_details_query = """
            query GetGenesById($nodeId: ID!) {
              geneSetByNodeId(nodeId: $nodeId) { genes { edges { node { symbol } } } }
            }
            """
            
            master_gene_list = set()
            for gene_set in top_20_sets:
                gene_set_info = gene_set.get('geneSetById')
                if not gene_set_info or not gene_set_info.get('nodeId'):
                    continue

                details_response = requests.post(url, json={
                    "operationName": "GetGenesById",
                    "query": gene_details_query, 
                    "variables": {"nodeId": gene_set_info['nodeId']}
                })
                details_response.raise_for_status()
                details_data = details_response.json()
                
                gene_edges = details_data.get("data",{}).get("geneSetByNodeId",{}).get("genes",{}).get("edges",[])
                for edge in gene_edges:
                    if symbol := edge.get('node', {}).get('symbol'):
                        master_gene_list.add(symbol)

            if not master_gene_list:
                print("No genes found in the top 20 sets.")
                return pd.DataFrame()

            # --- PHASE 3: Run GSEA on Aggregated Gene List ---
            gsea_results_df = self.run_gsea_enrichment(list(master_gene_list), disease_name)
            if gsea_results_df.empty:
                print("GSEA analysis did not produce results. Cannot calculate gene scores.")
                return pd.DataFrame()
            
            # --- PHASE 4: Filter GSEA Results (LLM Placeholder) ---
            top_gsea_terms = self._filter_gsea_with_llm(gsea_results_df, disease_name)
            if top_gsea_terms.empty:
                print("No GSEA terms remained after filtering. Cannot score genes.")
                return pd.DataFrame()
            
            print("\nTop 20 GSEA Terms selected for scoring:")
            print(top_gsea_terms[['Term', 'Adjusted P-value', 'similarity', 'Genes']])

            # --- PHASE 5: Calculate Gene Scores from Top Terms ---
            print("\nPHASE 3: Aggregating GSEA results and calculating final scores...")
            gene_evidence = collections.defaultdict(lambda: {'p_values': [], 'terms': set()})

            for _, row in top_gsea_terms.iterrows():
                p_val = row['Adjusted P-value']
                term = row['Term']
                genes_in_term = row['Genes'].split(';')
                for gene in genes_in_term:
                    gene_evidence[gene]['p_values'].append(p_val)
                    gene_evidence[gene]['terms'].add(term)

            # --- PHASE 6: Format Final DataFrame ---
            processed_data = []
            # Use a very small number to substitute for p-values of 0 to avoid log(0) errors
            min_p_value = 1e-323 

            for gene, evidence in gene_evidence.items():
                # Score is the -log10 of the best (minimum) p-value
                best_p = max(min(evidence['p_values']), min_p_value)
                score = -np.log10(best_p)
                
                processed_data.append({
                    "gene_symbol": gene,
                    "source": self.source_name,
                    "score": score,
                    "evidence": list(evidence['terms'])
                })

            if not processed_data:
                print("Could not generate final scored data.")
                return pd.DataFrame()

            final_df = pd.DataFrame(processed_data)
            return final_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            print(f"An API error occurred: {e}")
            return pd.DataFrame()
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Error processing data. This may be due to an unexpected API response. Error: {e}")
            return pd.DataFrame()

#summarize how the opentargets scoring works
class OpenTargets(DataSource):
    def __init__(self, source_name="OpenTargets"):
        super().__init__(source_name)

    def get_dsource(self):
        # Placeholder for fetching data from the OpenTargets
        return f"Data from {self.source_name}"

    
    def get_genes(self, disease_id: str) -> pd.DataFrame:
        """Queries OpenTargets for gene-disease associations."""
        print(f"\n[DEBUG:OpenTargets] Received disease_id: {disease_id}")
        # Normalize EFO ID to use colon for the API query
        normalized_efo_id = disease_id.replace(':', '_')
        print(f"[DEBUG:OpenTargets] Normalized EFO ID: {normalized_efo_id}")

        url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query associatedTargetsQuery($efoId: String!) {
          disease(efoId: $efoId) {
            associatedTargets {
              rows {
                target { approvedSymbol }
                score
              }
            }
          }
        }
        """
        try:
            response = requests.post(url, json={"query": query, "variables": {"efoId": normalized_efo_id}})
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[DEBUG:OpenTargets] OpenTargets query failed: {e}")
            return pd.DataFrame()

        associations = data.get('data', {}).get('disease', {}).get('associatedTargets', {}).get('rows', [])
        if not associations:
            print(f"[DEBUG:OpenTargets] No associations found for {normalized_efo_id}")
            return pd.DataFrame()
        
        print(f"[DEBUG:OpenTargets] Found {len(associations)} associations for {normalized_efo_id}")

        processed_data = [
            {
                "gene_symbol": row['target']['approvedSymbol'],
                "source": self.source_name,
                "score": row['score'],
                "evidence": [] 
            }
            for row in associations
        ]
        return pd.DataFrame(processed_data)