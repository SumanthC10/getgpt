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
from itertools import chain # Added import for chain

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


def source_to_url(src: str) -> str:
    """Return a hyperlink for a given source string."""
    if src.startswith("GSE"):
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={src}"
    if src.startswith("GCST"):          # GWAS Catalog study accession
        return f"https://www.ebi.ac.uk/gwas/studies/{src}"
    if src.lower().startswith("efo"):   # just in case you expose EFO IDs
        return f"https://www.ebi.ac.uk/ols/ontologies/efo/terms?short_form={src.replace(':','_')}"
    if src.lower() == "gwas catalog":
        return "https://www.ebi.ac.uk/gwas/"
    # …add any other data sources you use
    return "#"  

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
        try:
            # Normalize EFO ID to use colon
            normalized_efo_id = disease_id.replace(':', '_')
            # 1. Find all studies for the given disease ID
            studies_df = get_studies_by_efo_id(normalized_efo_id).studies
            if studies_df.empty:
                return pd.DataFrame()
        except Exception as e:
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
                        "source": [self.source_name],
                        "g_score": min(-np.log10(data['score']) /  -np.log10(5e-8), 1),
                        "evidence": data['evidence']
                    })
            except Exception as e:
                print(f"Warning: Could not process study {study_id}. Error: {e}")
                continue
        if not all_results:
            return pd.DataFrame()
        final_df = pd.DataFrame(all_results)
        # Define aggregation logic for grouping by gene symbol
        agg_logic = {
            'source': 'first',
            'g_score': 'mean',
            'evidence': lambda x: list(chain.from_iterable(x))
        }
        # Group by gene and aggregate
        aggregated_df = final_df.groupby('gene_symbol').agg(agg_logic).reset_index()

        return aggregated_df[['gene_symbol', 'source', 'g_score', 'evidence']].sort_values(by='g_score', ascending=False).reset_index(drop=True)



class RummaGEO(DataSource):
    def __init__(self, source_name="RummaGEO", lookup_file="data/efo_to_gse_lookup.json", efo_ontology_file="data/efo_ontology_terms.tsv"):
        super().__init__(source_name)
        self.graphql_url = "https://rummageo.com/graphql"
        self.headers = {"Content-Type": "application/json"}
        self.efo_ontology_file = efo_ontology_file
        try:
            with open(lookup_file, 'r') as f:
                self.efo_to_gse = json.load(f)
            print("[DEBUG:RummaGEO] Successfully loaded EFO to GSE lookup.")
        except FileNotFoundError:
            print(f"[DEBUG:RummaGEO] Error: EFO to GSE lookup file not found at {lookup_file}")
            self.efo_to_gse = {}

    def get_dsource(self):
        return f"Data from {self.source_name}"

    def _get_disease_name_from_efo_id(self, efo_id: str) -> str | None:
        """Finds the disease name for a given EFO ID from the ontology file."""
        normalized_efo_id = efo_id.replace('_', ':')
        try:
            with open(self.efo_ontology_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > 1 and parts[0] == normalized_efo_id:
                        return parts[1]
            return None
        except FileNotFoundError:
            print(f"[DEBUG:RummaGEO] Error: EFO ontology file not found at {self.efo_ontology_file}")
            return None

    def _fetch_genes_for_gse(self, gse_id: str) -> list:
        """Fetches genes for a single study given its GSE ID."""
        query = """
        query GeneSetGsesQuery($gse: String!) {
          geneSetGses(condition: {gse: $gse}) {
            edges {
              node {
                geneSetById {
                  genes {
                    edges {
                      node {
                        symbol
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        try:
            response = requests.post(self.graphql_url, json={
                "operationName": "GeneSetGsesQuery", "query": query, "variables": {"gse": gse_id}
            })
            response.raise_for_status()
            data = response.json()
            gse_edges = data.get("data", {}).get("geneSetGses", {}).get("edges", [])
            if not gse_edges: return []
            gene_set_by_id = gse_edges[0].get('node', {}).get('geneSetById', {})
            if not gene_set_by_id: return []
            gene_edges = gene_set_by_id.get("genes", {}).get("edges", [])
            return [edge['node']['symbol'] for edge in gene_edges if edge.get('node', {}).get('symbol')]
        except (requests.exceptions.RequestException, KeyError, TypeError, IndexError) as e:
            print(f"[DEBUG:RummaGEO] Could not fetch genes for {gse_id}. Error: {e}")
            return []

    def _run_gsea_enrichment(self, gene_list: list, disease_name: str) -> pd.DataFrame:
        """Performs GSEApy enrichment and filters results."""
        if not gene_list:
            print("[DEBUG:RummaGEO] GSEA enrichment called with an empty gene list.")
            return pd.DataFrame()
        print(f"\n[DEBUG:RummaGEO] Running GSEApy enrichment for '{disease_name}' with {len(gene_list)} genes...")
        try:
            enr_results = gp.enrichr(
                gene_list=gene_list, gene_sets='DisGeNET', organism='human', outdir=None, cutoff=1
            )
            if enr_results is None or enr_results.results.empty:
                print("[DEBUG:RummaGEO] GSEApy analysis returned no results.")
                return pd.DataFrame()
            
            results_df = enr_results.results
            print(results_df.sort_values(by='Adjusted P-value', ascending=True).head(10))
            print("[DEBUG:RummaGEO] GSEApy Analysis Complete. Filtering results...")

            # Filter using sentence transformer
            model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
            disease_embedding = model.encode(disease_name)
            term_embeddings = model.encode(results_df['Term'].tolist())
            similarities = cos_sim(disease_embedding, term_embeddings)
            results_df['similarity'] = similarities[0]
            
            return results_df.sort_values(by='similarity', ascending=False).head(20)
        except Exception as e:
            print(f"[DEBUG:RummaGEO] An error occurred during GSEApy analysis: {e}")
            return pd.DataFrame()

    def get_genes(self, disease_id: str) -> pd.DataFrame:
        """
        Combines gene fetching from RummaGEO with GSEApy enrichment analysis to score genes.
        """
        print(f"\n[DEBUG:RummaGEO] Received disease_id: {disease_id}")
        
        # 1. Fetch master gene list from RummaGEO
        studies_from_lookup = self.efo_to_gse.get(disease_id, [])
        if not studies_from_lookup:
            print(f"[DEBUG:RummaGEO] No studies found for {disease_id} in the lookup file.")
            return pd.DataFrame()

        gene_to_gses = collections.defaultdict(list)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_gse = {executor.submit(self._fetch_genes_for_gse, study['gse_id']): study['gse_id'] for study in studies_from_lookup}
            for future in as_completed(future_to_gse):
                gse_id = future_to_gse[future]
                genes = future.result()
                for gene in genes:
                    gene_to_gses[gene].append(gse_id)

        master_gene_list = list(gene_to_gses.keys())

        if not master_gene_list:
            print("[DEBUG:RummaGEO] No genes found for any of the target studies.")
            return pd.DataFrame()
        
        # 2. Get disease name for GSEA
        disease_name = self._get_disease_name_from_efo_id(disease_id)
        if not disease_name:
            print(f"[DEBUG:RummaGEO] Could not find disease name for EFO ID {disease_id}. Aborting GSEA.")
            return pd.DataFrame()

        # 3. Run GSEA and get top terms
        top_gsea_terms = self._run_gsea_enrichment(list(master_gene_list), disease_name)
        if top_gsea_terms.empty:
            print("[DEBUG:RummaGEO] No GSEA terms remained after filtering. Cannot score genes.")
            return pd.DataFrame()

        # 4. Calculate gene scores from top terms
        gene_evidence = collections.defaultdict(lambda: {'p_values': [], 'terms': set()})
        for _, row in top_gsea_terms.iterrows():
            p_val = row['P-value']
            print(f"[DEBUG:RummaGEO] Processing term: {row['Term']} with p-value: {p_val}")
            term = row['Term']
            genes_in_term = row['Genes'].split(';')
            for gene in genes_in_term:
                if gene in master_gene_list: # Ensure we only score genes from our initial list
                    gene_evidence[gene]['p_values'].append(p_val)
                    gene_evidence[gene]['terms'].add(term)

        # 5. Format final DataFrame
        processed_data = []
        for gene, evidence in gene_evidence.items():
            # Find the best (lowest) p-value for the gene
            best_p = min(evidence['p_values'])
            processed_data.append({
                "gene_symbol": gene,
                "source": [self.source_name],
                "e_score": min((-np.log10(best_p) / (-np.log10(5e-8))), 1),
                "evidence": sorted(list(set(gene_to_gses.get(gene, []))))
            })
        if not processed_data:
            print("[DEBUG:RummaGEO] Could not generate final scored data.")
            return pd.DataFrame()
        final_df = pd.DataFrame(processed_data)

        return final_df[['gene_symbol', 'source', 'e_score', 'evidence']].sort_values(by='e_score', ascending=False).reset_index(drop=True)

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
                id
                name
                associatedTargets {
                count
                rows {
                    target {
                    id
                    approvedSymbol
                    }
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
                "source": [self.source_name],
                "t_score": row['score'],
                "evidence": []
            }
            for row in associations
        ]
        final_df = pd.DataFrame(processed_data)
        return final_df[['gene_symbol', 'source', 't_score', 'evidence']].sort_values(by='t_score', ascending=False).reset_index(drop=True)