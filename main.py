
import streamlit as st
import pandas as pd
import json
import time
import logging
import concurrent.futures
import os, time, io, shutil, tempfile, zipfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pager_backend import pager_analysis


# Import functions from g2d_utils.py
from g2d_utils import (
    load_g2d_data,
    load_mesh_mapping,
    load_mesh_descriptors,
    compute_overlap_by_mesh,
    load_openai_llm,
    get_explanation_sections,
    get_disease_summary
)

# Additional imports for your other functionalities
from opentargets_api import (
    get_associated_targets, 
    get_studies, 
    get_variants_for_study,
    test_opentargets_api,
    test_opentargets_genetics_api
)
from efo_conversion import get_mesh_id_from_file, get_oxo_mapping_from_mesh
from pubmed import query_pubmed_for_abstracts
from gene_extraction import extract_gene_names, extract_genes_with_chatgpt
from pdf_processing import read_pdf_file
from semantic_search import load_mesh_data, search_mesh, precompute_embeddings

def run_pager(disease: str, genes: str, top_n: str):
    # returns (adv_df, pag_files:dict, zip_bytes:bytes)
    return pager_analysis(disease, genes, top_n)

# Logging configuration
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ChatOpenAI for gene extraction (if needed in other parts)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4.1",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.1,
    max_tokens=4096
)

pdf_llm = ChatOpenAI(
    model_name="gpt-4.1",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.0,
    max_tokens=4096
)

def compute_sources(row):
    """
    Computes a combined source code based on available columns.
    T: Drug Targets (OpenTargets)
    E: Differentially Expressed Genes (from PubMed abstracts)
    G: Genetics / DisGeNET (shows rsid and accession ID)
    """
    sources = []
    if row.get("rsid", "N/A") != "N/A":
        sources.append("G")
    if row.get("Found in PubMed", "No") == "Yes":
        sources.append("E")
    if row.get("OpenTargets Score", "N/A") != "N/A" or row.get("Found in DisGeNET", "No") == "Yes":
        sources.append("T")
    return "+".join(sources) if sources else "-"

def test_opentargets_api_wrapper():
    try:
        result = test_opentargets_api()
        return result
    except Exception as e:
        logger.exception("Error testing OpenTargets API")
        return f"Error testing OpenTargets API: {str(e)}"

def test_opentargets_genetics_api_wrapper():
    try:
        result = test_opentargets_genetics_api()
        return result
    except Exception as e:
        logger.exception("Error testing OpenTargets Genetics API")
        return f"Error testing OpenTargets Genetics API: {str(e)}"

def query_opentargets(disease_name):
    try:
        # Convert disease name to a MeSH ID and then to a EFO ID.
        mesh_id = get_mesh_id_from_file(disease_name)
        efo_id_raw = get_oxo_mapping_from_mesh(mesh_id)
        # Convert from "EFO:XXXX" to "EFO_XXXX" if required by OpenTargets.
        efo_id = efo_id_raw.replace(":", "_")
        
        result = ""
        
        # Associated Targets.
        try:
            associated_targets = get_associated_targets(efo_id)
            opentargets_genes = {}
            if not associated_targets:
                result += "No associated targets found in OpenTargets.\n"
            else:
                for target in associated_targets:
                    opentargets_genes[target["approvedSymbol"]] = target["score"]
        except Exception as e:
            logger.error(f"Error fetching associated targets: {e}")
            opentargets_genes = {}
            result += "Error fetching associated targets from OpenTargets.\n"
        
        # Genetics Studies.
        try:
            studies = get_studies(efo_id)
        except Exception as e:
            logger.error(f"Error fetching studies: {e}")
            studies = []
            result += "Error fetching genetic studies.\n"
        
        # Process variant associations via bestGenes.
        genetics_genes = {}
        for study in studies:
            study_id = study["id"]
            try:
                associations = get_variants_for_study(study_id)
                for assoc in associations:
                    variant = assoc.get("variant", {})
                    if not variant:
                        continue
                    variant_id = variant.get("id", "")
                    rsid = variant.get("rsId", "")
                    if not rsid:
                        rsid = variant_id
                    # Use bestGenes field to determine the best gene for this variant.
                    best_genes = assoc.get("bestGenes", [])
                    if not best_genes:
                        continue
                    # Choose the gene with the highest score.
                    best_gene = max(best_genes, key=lambda x: float(x.get("score", 0)))
                    gene_info = best_gene.get("gene")
                    # Print the variant and gene info using st.write.
                    if gene_info:
                        gene_symbol = gene_info.get("symbol")
                        if gene_symbol:
                            entry = genetics_genes.get(gene_symbol, {
                                "rsid": [],
                                "accession": [],
                                "study": []
                            })

                            # only add if not already present
                            if rsid and rsid not in entry["rsid"]:
                                entry["rsid"].append(rsid)
                            if variant_id and variant_id not in entry["accession"]:
                                entry["accession"].append(variant_id)
                            if study_id and study_id not in entry["study"]:
                                entry["study"].append(study_id)

                            genetics_genes[gene_symbol] = entry
            except Exception as e:
                logger.error(f"Error fetching variant associations for study {study['id']}: {e}")
        if not genetics_genes:
            result += "No OpenTargets Genetics associations found.\n"
        
        # PubMed abstracts.
        try:
            abstracts, pubmed_gene_pmids = query_pubmed_for_abstracts(disease_name, llm=llm)
            if not pubmed_gene_pmids:
                result += "No genes extracted from PubMed abstracts.\n"
            expression_genes = set(pubmed_gene_pmids.keys()) if pubmed_gene_pmids else set()
        except Exception as e:
            logger.error(f"Error querying PubMed abstracts: {e}")
            result += "Error querying PubMed abstracts.\n"
            pubmed_gene_pmids = {}
            expression_genes = set()
        
        # Build the union of genes.
        all_genes = set(opentargets_genes.keys()) | expression_genes | set(genetics_genes.keys())
        
        # Only add "no data" messages for a given source; if data exists, do not print details.
        if not expression_genes:
            result += "No genes extracted from PubMed abstracts.\n"
        if not opentargets_genes:
            result += "No genes found from OpenTargets.\n"
        if not genetics_genes:
            result += "No genetics associations found.\n"
        
        gene_data = []
        for gene in sorted(all_genes):
            get_codes = []
            if gene in genetics_genes:
                get_codes.append("G")
            if gene in expression_genes:
                get_codes.append("E")
            if gene in opentargets_genes:
                get_codes.append("T")
            get_str = "+".join(get_codes) if get_codes else "-"
            gene_data.append({
                "Gene": gene,
                "GET": get_str,
                "OpenTargets Score": opentargets_genes.get(gene, "N/A"),
                "Found in PubMed": "Yes" if gene in expression_genes else "No",
                "PMID": ", ".join(pubmed_gene_pmids[gene]) if gene in pubmed_gene_pmids else "N/A",
                "rsid": ", ".join(genetics_genes[gene]["rsid"]) if gene in genetics_genes else "N/A",
                "Study ID": ", ".join(genetics_genes[gene]["study"]) if gene in genetics_genes else "N/A"
            })
        
        return result, gene_data

    except Exception as e:
        logger.exception(f"Error in query_opentargets: {str(e)}")
        return f"An error occurred while querying OpenTargets and extracting genes: {str(e)}", None

def toggle_run():
    st.session_state["is_running"] = not st.session_state.get("is_running", False)
    if not st.session_state["is_running"]:
        st.session_state["stop_query"] = True

def main():
    for key, default in {
    "pager_download_dir": None,
    "pager_results_ready": False,
    "pager_adv_csv"     : None,
    "pager_pag_files"   : {},
    "pager_zip_buf"     : None,
    "pager_disease"     : None,
    "pager_num_pags"    : "All",
        }.items():
        st.session_state.setdefault(key, default)

    st.set_page_config(page_title="GetGPT", layout="wide")
    if "saved_summaries" not in st.session_state:
        st.session_state["saved_summaries"] = {}
    st.title("GetGPT")
    
    # Initialize session state variables.
    for key in ["stop_query", "is_running", "gene_list", "gene_table", "results_df", "base_disease", "selected_mesh_d2g", "selected_mesh_d2g_code"]:
        if key not in st.session_state:
            st.session_state[key] = None if key in ["gene_table", "results_df", "base_disease", "selected_mesh_d2g", "selected_mesh_d2g_code"] else False
    if st.session_state["gene_list"] is None:
        st.session_state["gene_list"] = []

    # Load MeSH data.
    mesh_preferred_terms, mesh_alt_terms, preferred_embeddings, alt_embeddings = load_mesh_data()
    mesh_mapping = load_mesh_mapping()

    tab_gt, tab_pg, tab_llm, tab_api = st.tabs(["GetGPT", "PAGER", "LLM Summaries", "API Testing"])    
    # --- First Tab: D2G Associations ---
    with tab_gt:
        st.header("Enter a Disease")
        from streamlit_searchbox import st_searchbox
        selected_mesh_d2g = st_searchbox(
            lambda s: search_mesh(s, mesh_preferred_terms, mesh_alt_terms, preferred_embeddings, alt_embeddings, 
                                    top_n=5, weight_semantic=0.7, weight_fuzzy=0.3, 
                                    fuzzy_weight_pref=0.5, fuzzy_weight_alt=0.5),
            placeholder="Type a Disease...",
            key="mesh_search"
        )
        if selected_mesh_d2g:
            st.write(f"Selected Disease: **{selected_mesh_d2g}**")
            if st.session_state.get("selected_mesh_d2g") != selected_mesh_d2g:
                st.session_state["gene_list"] = []
                st.session_state["gene_table"] = None
                st.session_state["results_df"] = None
                st.session_state["base_disease"] = selected_mesh_d2g
                # --- reset PAGER state too! ---
                for k in [
                    "pager_download_dir",
                    "pager_adv_df",    
                    "pager_files", 
                    "pager_zip_bytes", 
                ]:
                    st.session_state.pop(k, None)
            st.session_state["selected_mesh_d2g"] = selected_mesh_d2g

        pdf_files = st.file_uploader("Drag and drop your PDF file for gene extraction here", 
                                     type="pdf", accept_multiple_files=True)
        
        toggle_label = "Stop Query" if st.session_state["is_running"] else "Run Analysis"
        st.button(toggle_label, on_click=toggle_run, key="toggle_button")
        
        if st.session_state["is_running"]:
            if not selected_mesh_d2g or selected_mesh_d2g not in mesh_preferred_terms:
                st.error("Please select a valid MeSH term from the suggestions before running the analysis.")
            else:
                st.session_state["base_disease"] = selected_mesh_d2g
                disease_name = selected_mesh_d2g
                with st.spinner("Querying OpenTargets, Genetics and PubMed abstracts..."):
                    result_text, gene_data = query_opentargets(disease_name)
                if gene_data:
                    df_genes = pd.DataFrame(gene_data)
                    df_genes.index = range(1, len(df_genes) + 1)
                    st.session_state["gene_list"] = sorted(set(df_genes["Gene"].tolist()))
                    st.session_state["gene_table"] = df_genes.copy()
                else:
                    st.session_state["gene_table"] = pd.DataFrame(columns=["Gene", "OpenTargets Score", "Found in PubMed"])
                
                if pdf_files is not None:
                    pdf_genes_set = set()
                    for file_index, pdf_file in enumerate(pdf_files, start=1):
                        full_text, _ = read_pdf_file(pdf_file)
                        with st.spinner(f"Extracting genes from PDF file {file_index}…"):
                            # now rely on your smarter helper to chunk if needed
                            raw_genes = extract_genes_with_chatgpt(
                                full_text,
                                context_label=f"PDF file {file_index}",
                                llm=pdf_llm
                            )
                        # whitelist‐filter
                        allowed = set(pd.read_csv("data/gene_names.txt")["symbol"].str.strip())
                        filtered = [g for g in raw_genes if g in allowed]
                        pdf_genes_set |= set(filtered)

                    if pdf_genes_set:
                        df_genes = st.session_state["gene_table"]
                        df_genes["Found in PDF"] = df_genes["Gene"].apply(lambda x: "Yes" if x in pdf_genes_set else "No")
                        missing = pdf_genes_set - set(df_genes["Gene"])
                        if missing:
                            df_add = pd.DataFrame([{
                                "Gene": g,
                                "OpenTargets Score": "N/A",
                                "Found in PubMed": "No",
                                "Found in PDF": "Yes"
                            } for g in missing])
                            df_genes = pd.concat([df_genes, df_add], ignore_index=True)
                            df_genes.index = range(1, len(df_genes) + 1)
                        st.session_state["gene_table"] = df_genes.copy()
                                                
                    if pdf_genes_set:
                        df_genes = st.session_state["gene_table"]
                        df_genes["Found in PDF"] = df_genes["Gene"].apply(lambda x: "Yes" if x in pdf_genes_set else "No")
                        missing_genes = pdf_genes_set - set(df_genes["Gene"])
                        if missing_genes:
                            additional_rows = [{
                                "Gene": gene,
                                "OpenTargets Score": "N/A",
                                "Found in PubMed": "No",
                                "Found in PDF": "Yes"
                            } for gene in missing_genes]
                            df_additional = pd.DataFrame(additional_rows)
                            df_genes = pd.concat([df_genes, df_additional], ignore_index=True)
                            df_genes.index = range(1, len(df_genes) + 1)
                        st.session_state["gene_table"] = df_genes.copy()
                
                try:
                    allowed_genes_df = pd.read_csv("data/gene_names.txt")
                    allowed_genes = set(allowed_genes_df["symbol"].str.strip())
                    df_genes = st.session_state["gene_table"]
                    df_genes = df_genes[df_genes["Gene"].isin(allowed_genes)]
                    df_genes.index = range(1, len(df_genes) + 1)
                    st.session_state["gene_table"] = df_genes.copy()
                    st.session_state["gene_list"] = sorted(set(df_genes["Gene"].tolist()))
                except Exception as e:
                    st.error(f"Error reading or filtering by gene_names.txt: {e}")
                
                try:
                    disgenet_df = pd.read_csv("data/d2g_final_new.csv")
                    if "Unnamed: 0" in disgenet_df.columns:
                        disgenet_df = disgenet_df.drop(columns=["Unnamed: 0"])
                    selected_disease_lower = st.session_state["selected_mesh_d2g"].strip().lower()
                    disgenet_filtered = disgenet_df[disgenet_df["DISEASE_NAME"].str.lower().str.contains(selected_disease_lower)]
                    gene_list_set = set(st.session_state["gene_list"])
                    disgenet_filtered = disgenet_filtered[disgenet_filtered["gene"].isin(gene_list_set)]
                    df_genes = st.session_state["gene_table"]
                    df_genes["Found in DisGeNET"] = df_genes["Gene"].apply(
                        lambda x: "Yes" if x in set(disgenet_filtered["gene"]) else "No"
                    )
                except Exception as e:
                    st.error(f"Error loading or processing DisGeNET data from CSV: {e}")
                
                df_genes.fillna("N/A", inplace=True)
                st.session_state["gene_table"] = df_genes.copy()
                
                if st.session_state.get("gene_list"):
                    base_disease = st.session_state["selected_mesh_d2g"]
                    descriptors_df = load_mesh_descriptors()
                    df = load_g2d_data()
                    mapping_df = load_mesh_mapping()
                    overlap_results = compute_overlap_by_mesh(df, st.session_state["gene_list"], mapping_df, min_genes=3, initial_condition=base_disease)
                    if not overlap_results.empty:
                        if "MeSH_code" in overlap_results.columns:
                            overlap_results = overlap_results.merge(
                                descriptors_df[["DescriptorUI", "PreferredTerm"]],
                                left_on="MeSH_code",
                                right_on="DescriptorUI",
                                how="left"
                            ).rename(columns={"MeSH_code": "MeSH_Code"})
                        elif "MeSH_Code" in overlap_results.columns:
                            overlap_results = overlap_results.merge(
                                descriptors_df[["DescriptorUI", "PreferredTerm"]],
                                left_on="MeSH_Code",
                                right_on="DescriptorUI",
                                how="left"
                            )
                        overlap_results["Disease_Name"] = overlap_results["PreferredTerm"].fillna(overlap_results["MeSH_Code"])
                        for col in ["PreferredTerm", "DescriptorUI", "Original_MeSH"]:
                            if col in overlap_results.columns:
                                overlap_results = overlap_results.drop(columns=[col])
                        overlap_results = overlap_results[
                            overlap_results["Disease_Name"].str.strip().str.lower() != base_disease.strip().lower()
                        ]
                        overlap_results = overlap_results[overlap_results["p_value"] < 0.05]
                        overlap_results = overlap_results.sort_values("p_value").head(15)
                        overlap_results.index = range(1, len(overlap_results) + 1)
                        st.session_state["results_df"] = overlap_results
                    else:
                        st.error("No overlapping diseases found with p < 0.05.")
                
                st.session_state["is_running"] = False
        
        # -----------------------------
        # Display the gene table with the GET column as the second column
        # -----------------------------
        if st.session_state.get("gene_table") is not None and st.session_state.get("gene_list"):
            df_display = st.session_state["gene_table"].copy()
            desired_order = ["Gene", "GET", "OpenTargets Score", "Found in PubMed", "PMID", "rsid", "accession", "Study ID", "Found in PDF"]
            if "GET" not in df_display.columns:
                def compute_GET(row):
                    codes = []
                    if row.get("rsid", "N/A") != "N/A":
                        codes.append("G")
                    if row.get("Found in PubMed", "No") == "Yes":
                        codes.append("E")
                    if row.get("OpenTargets Score", "N/A") != "N/A":
                        codes.append("T")
                    return "+".join(codes) if codes else "-"
                df_display["GET"] = df_display.apply(compute_GET, axis=1)
            df_display = df_display[[col for col in desired_order if col in df_display.columns]]
            disease_filename = (st.session_state["selected_mesh_d2g"].replace(" ", "_")
                                if st.session_state.get("selected_mesh_d2g") else "analysis")
            st.subheader("Geneset")
            st.dataframe(df_display, width = 1000)
            geneset_csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download Geneset CSV",
                data=geneset_csv,
                file_name=f"{disease_filename}_gene_analysis.csv",
                mime="text/csv",
            )
            st.markdown("""
            **Legend:**
            - **G**: Genetic Variants (OpenTargets Genetics)
            - **E**: Differentially Expressed Genes (PubMed abstracts) 
            - **T**: Drug Targets (OpenTargets)
            """, unsafe_allow_html=True)
            
            st.subheader("Top Overlapping Diseases From DisGeNET")
            if st.session_state.get("results_df") is not None:
                cols = [col for col in ["Disease_Name", "Overlap_Count", "p_value", "MeSH_Code"]
                        if col in st.session_state["results_df"].columns]
                st.dataframe(st.session_state["results_df"][cols].style.format({"p_value": "{:.2e}"}))
                overlap_csv = st.session_state["results_df"].to_csv(index=False)
                st.download_button(
                    label="Download Overlap CSV",
                    data=overlap_csv,
                    file_name=f"{disease_filename}_disease_overlap.csv",
                    mime="text/csv",
                )
            else:
                st.info("Overlap analysis not available yet.")
    
    # --- Sidebar: LLM Explanation ---
    if st.session_state.get("results_df") is not None and not st.session_state["results_df"].empty:
        with st.sidebar:
            st.subheader("LLM Explanation")
            descriptors_df = load_mesh_descriptors()
            disease_list = st.session_state["results_df"]["Disease_Name"].tolist()
            selected_disease2 = st.selectbox("Select Disease for Explanation", disease_list, key="explanation_disease2")
            if st.button("Get Explanation", key="get_explanation"):
                spinner_placeholder = st.empty()
                spinner_placeholder.info("Getting explanation, please wait...")
                try:
                    base_disease_code = st.session_state["base_disease"]
                    base_row = descriptors_df[descriptors_df["DescriptorUI"] == base_disease_code]
                    base_disease_name = base_row.iloc[0]["PreferredTerm"] if not base_row.empty else str(base_disease_code)
                    overlapping_genes1 = st.session_state["gene_list"]
                    row2 = st.session_state["results_df"][st.session_state["results_df"]["Disease_Name"] == selected_disease2].iloc[0]
                    overlapping_genes2 = row2["Overlapping_Genes"]
                    llm_instance = load_openai_llm()
                    sections = get_explanation_sections(
                        base_disease_name, overlapping_genes1,
                        selected_disease2, overlapping_genes2,
                        llm_instance
                    )
                    st.session_state["explanation_sections"] = sections
                    st.session_state["llm_explanations"].append({
                        "initial": base_disease_name,
                        "target": selected_disease2,
                        "sections": sections
                    })
                    st.session_state["saved_summaries"][selected_disease2] = sections
                except Exception as e:
                    logger.exception(f"Explanation query failed: {e}")
                    st.error(f"Explanation query failed: {e}")
                finally:
                    spinner_placeholder.empty()

            if st.session_state.get("explanation_sections"):
                st.subheader("Explanation of Overlapping Genes & Disease Associations")
                for header, content in st.session_state["explanation_sections"].items():
                    st.markdown("#### " + header)
                    st.write(content)
    
    # ─── TAB 2: PAGER ──────────────────────────────────────────
    with tab_pg:
        st.header("PAGER Gene Set Enrichment")

        genes = st.session_state.get("gene_list", [])
        if not genes:
            st.warning("No gene list available. Run the GetGPT tab first.")
        else:
            disease_pg = st.session_state["selected_mesh_d2g"]
            num_pags_pg = st.selectbox(
                "Number of PAGs to download:",
                ["5","10","15","25","50","All"],
                index=5,
                key="pager_num_pags"
            )

            if st.button("Analyze on PAGER", key="pager_run"):
                with st.spinner("Running PAGER analysis…"):
                    try:
                        adv_df, pag_files, zip_bytes = run_pager(
                            disease_pg,
                            "\n".join(genes),
                            num_pags_pg
                        )
                        adv_df.index = range(1, len(adv_df) + 1)
                        st.session_state["pager_adv_df"]   = adv_df
                        st.session_state["pager_files"]    = pag_files
                        st.session_state["pager_zip_bytes"] = zip_bytes
                    except Exception as e:
                        st.error(f"PAGER failed: {e}")
                        st.stop()

            if "pager_adv_df" in st.session_state:
                tab_summary, tab_individual = st.tabs(["Results", "Individual PAGs"])
                with tab_summary:
                    # Summary
                    st.subheader("All Associated PAGs")
                    st.dataframe(st.session_state["pager_adv_df"])
                    n = st.session_state.get("pager_num_pags", "All")
                    st.subheader(f"Download PAG List and {n} PAGs Information")
                    st.download_button(
                        "Download ZIP File",
                        data=st.session_state["pager_zip_bytes"],
                        file_name=f"{disease_pg.replace(' ','_')}.zip",
                        mime="application/zip"
                    )
                with tab_individual:
                    st.subheader("Individual PAGs")
                    adv_df = st.session_state["pager_adv_df"]
                    for pid, path in st.session_state["pager_files"].items():
                        # look up the human‑readable name in the advanced‑search df
                        try:
                            pag_name = (
                                adv_df.loc[adv_df["PAG ID"] == pid, "Name"]
                                .iloc[0]
                            )
                        except Exception:
                            pag_name = pid

                        # show both name and ID in the expander header
                        with st.expander(f"{pag_name} ({pid})"):
                            st.markdown(f"**PAG Name:** {pag_name}  \n**PAG ID:** {pid}")

                            df = pd.read_csv(path)
                            df.index = range(1, len(df) + 1)
                            st.dataframe(df, width=1500)

                            with open(path, "rb") as f:
                                st.download_button(
                                    "Download CSV",
                                    data=f.read(),
                                    file_name=os.path.basename(path),
                                    mime="text/csv"
                                )
    
    # ─── TAB 3: LLM Summaries ──────────────────────────────────
    with tab_llm:
        st.header("LLM Summaries")

        # 1) Ensure our list exists in session_state
        if "llm_explanations" not in st.session_state:
            st.session_state["llm_explanations"] = []

        # 2) Render
        exps = st.session_state["llm_explanations"]
        if not exps:
            st.info("No explanations generated yet. Use the Get Explanation button in the sidebar.")
        else:
            for exp in reversed(exps):
                title = f"{exp['initial']} → {exp['target']}"
                with st.expander(title, expanded=False):
                    for header, content in exp["sections"].items():
                        st.markdown(f"#### {header}")
                        st.write(content)

            # 3) Download everything
            payload = [
                {
                    "initial": e["initial"],
                    "target":  e["target"],
                    "sections": e["sections"]
                }
                for e in exps
            ]
            st.download_button(
                "Download All Explanations as JSON",
                data=json.dumps(payload, indent=2),
                file_name="llm_explanations.json",
                mime="application/json"
            )

    # ─── TAB 4: API Testing ───────────────────────────────────
    with tab_api:
        st.header("API Test")
        if st.button("Test OpenTargets and Genetics API"):
            with st.spinner("Testing API..."):
                result = test_opentargets_api_wrapper()
                result_genetics = test_opentargets_genetics_api_wrapper()
            st.markdown(result, unsafe_allow_html=True)
            st.markdown(result_genetics, unsafe_allow_html=True)

if __name__ == "__main__":
    main()