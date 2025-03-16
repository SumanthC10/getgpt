import io
import logging
import requests
from Bio import Medline

logger = logging.getLogger(__name__)

def query_pubmed_for_abstracts(disease_name, max_results=5):
    try:
        # Step 1: Use ESearch to get PubMed IDs in JSON format
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": f"{disease_name} differentially expressed genes",
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json"
        }
        esearch_resp = requests.get(esearch_url, params=esearch_params, verify=False)
        esearch_resp.raise_for_status()
        search_results = esearch_resp.json()
        id_list = search_results.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return "No relevant PubMed articles found."
        
        # Step 2: Use EFetch to get article details in MEDLINE text format
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "rettype": "medline",
            "retmode": "text"
        }
        efetch_resp = requests.get(efetch_url, params=efetch_params, verify=False)
        efetch_resp.raise_for_status()
        medline_data = efetch_resp.text
        
        # Step 3: Parse the MEDLINE data using Bio.Medline
        handle = io.StringIO(medline_data)
        articles = list(Medline.parse(handle))
        abstracts = []
        for article in articles:
            title = article.get('TI', 'N/A')
            abstract = article.get('AB', 'N/A')
            abstracts.append(f"Title: {title}\n\nAbstract: {abstract}\n\n")
        return "\n".join(abstracts)
    except Exception as e:
        logger.exception("Error querying PubMed: %s", str(e))
        return f"An error occurred while querying PubMed: {str(e)}"