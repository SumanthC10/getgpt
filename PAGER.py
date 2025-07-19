import os
import time
import tempfile
import pandas as pd
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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