import os, time, io, tempfile, zipfile, concurrent.futures
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def _build_opts(folder):
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-features=DownloadBubble,DownloadBubbleV2")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_experimental_option("prefs", {
        "download.default_directory": folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    })
    return opts

def _wait_for_new_file(existing, folder, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        diff = set(os.listdir(folder)) - existing
        if diff:
            return diff.pop()
        time.sleep(0.5)
    return None

def _process_pag(pag_id, download_dir, base_url):
    pag_folder = os.path.join(download_dir, pag_id)
    os.makedirs(pag_folder, exist_ok=True)
    driver = webdriver.Chrome(service=Service(), options=_build_opts(pag_folder))
    try:
        driver.get(base_url + pag_id)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Download As']"))
        ).click()
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.dt-button.buttons-csv.buttons-html5"))
        ).click()
        new_file = _wait_for_new_file(set(), pag_folder, timeout=20)
    finally:
        driver.quit()
    if not new_file:
        return None
    src = os.path.join(pag_folder, new_file)
    dst = os.path.join(download_dir, f"{pag_id}_genes.csv")
    os.replace(src, dst)
    return pag_id, dst

def pager_analysis(disease_name: str, gene_input: str, num_pags: str = "All"):
    """
    Runs PAGER search+downloads. Returns:
      - adv_df: pd.DataFrame of the advanced‐search CSV
      - pag_files: dict[PAG_ID -> filepath]
      - zip_bytes: bytes of the ZIP archive
    """
    download_dir = tempfile.mkdtemp(prefix="pager_dl_")
    
    # 1) advanced search
    driver = webdriver.Chrome(service=Service(), options=_build_opts(download_dir))
    try:
        driver.get("https://discovery.informatics.uab.edu/PAGER/index.php/search/advanced")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "gene_list")))
        box = driver.find_element(By.ID, "gene_list")
        box.clear(); box.send_keys(gene_input)
        driver.find_element(By.ID, "filter_basic").click()
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Download As']"))
            ).click()
        except: pass
        existing = set(os.listdir(download_dir))
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.buttons-csv"))
        ).click()
        file_name = _wait_for_new_file(existing, download_dir, timeout=15)
        if not file_name:
            raise RuntimeError("Advanced-search CSV did not download.")
        adv_path = os.path.join(download_dir, file_name)
    finally:
        driver.quit()
    
    adv_df = pd.read_csv(adv_path)
    if not {"p-value","PAG ID"}.issubset(adv_df.columns):
        raise RuntimeError("Advanced CSV missing expected columns.")

    pag_ids = adv_df.sort_values("p-value")["PAG ID"].tolist()
    if num_pags != "All":
        pag_ids = pag_ids[:int(num_pags)]

    # 2) parallel PAG downloads
    base_url = "https://discovery.informatics.uab.edu/PAGER/index.php/geneset/view/"
    pag_files = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_pag, pid, download_dir, base_url): pid for pid in pag_ids}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                pag_files[res[0]] = res[1]

    if not pag_files:
        raise RuntimeError("No PAG files downloaded.")

    # 3) make ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(adv_path, arcname=f"{disease_name.replace(' ','_')}_pag_list.csv")
        for pid, path in pag_files.items():
            z.write(path, arcname=os.path.basename(path))
    buf.seek(0)

    return adv_df, pag_files, buf.read()