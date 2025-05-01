import pandas as pd
from scipy.stats import hypergeom
import re
import logging
from langchain_openai import ChatOpenAI
import streamlit as st
import torch
import concurrent.futures

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths (these should be defined in main.py or passed as parameters)
DATA_FILE = "data/d2g_final_new.csv"           # Gene-to-disease data; expected columns include 'gene', 'UMLS', and 'DISEASE_SEMANTIC_TYPE'
MESH_MAPPING_FILE = "data/mesh_umls_final.csv"   # Mapping file with columns: 'umls_code', 'mesh_code'
DESCRIPTORS_FILE = "data/mesh_terms.csv"         # File with columns: DescriptorUI, PreferredTerm, SearchTerms

# ======= Data Loading Functions =======
def load_g2d_data():
    """
    Load the gene-to-disease CSV file.
    Returns:
        DataFrame: Gene-to-disease data.
    """
    return pd.read_csv(DATA_FILE)


def load_mesh_mapping():
    """
    Load the MeSH mapping CSV file.
    Returns:
        DataFrame: Mapping data between UMLS codes and MeSH codes.
    """
    return pd.read_csv(MESH_MAPPING_FILE)


def load_mesh_descriptors():
    """
    Load the MeSH descriptors CSV file.
    Returns:
        DataFrame: Descriptors containing PreferredTerm and other details.
    """
    return pd.read_csv(DESCRIPTORS_FILE)


# ======= Analysis Function (Aggregated by MeSH Code) =======
def compute_overlap_by_mesh(df, input_genes, mapping_df, min_genes=3, initial_condition=None):
    """
    Groups the gene-to-disease data (filtered by semantic type) by the mapped MeSH code,
    computes the hypergeometric test, and returns a DataFrame with aggregated gene counts,
    overlap counts, p-values, and the list of overlapping genes.
    
    Args:
        df (DataFrame): Gene-to-disease data.
        input_genes (list): List of input genes.
        mapping_df (DataFrame): Mapping of UMLS to MeSH codes.
        min_genes (int): Minimum gene count threshold.
        initial_condition (str): The MeSH code to remove from the results.
        
    Returns:
        DataFrame: Contains original MeSH code, aggregated gene count, overlap count,
                   hypergeometric p-value, and list of overlapping genes.
    """
    # Filter to keep only disease or syndrome records.
    df_filtered = df[df["DISEASE_SEMANTIC_TYPE"] == "Disease or Syndrome"]
    background_genes = set(df_filtered["gene"].dropna())
    N = len(background_genes)
    m = len(input_genes)
    
    # Merge with mapping to get the MeSH codes.
    merged_df = df_filtered.merge(mapping_df, left_on="UMLS", right_on="umls_code", how="inner")
    
    # Group by the MeSH code.
    grouped = merged_df.groupby("mesh_code")["gene"].apply(lambda genes: set(genes.dropna())).reset_index()
    
    # Remove the row corresponding to the initial_condition mesh code.
    if initial_condition is not None:
        grouped = grouped[grouped["mesh_code"] != initial_condition]
    
    results = []
    for _, row in grouped.iterrows():
        mesh_code = row["mesh_code"]
        gene_set = row["gene"]
        K = len(gene_set)
        if K < min_genes:
            continue
        overlapping_genes = set(input_genes).intersection(gene_set)
        k = len(overlapping_genes)
        p_value = hypergeom.sf(k - 1, N, K, m)
        results.append({
            "MeSH_code": mesh_code,  # Original MeSH id
            "Total_Aggregated_Genes": K,
            "Overlap_Count": k,
            "p_value": p_value,
            "Overlapping_Genes": list(overlapping_genes)
        })
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("p_value")
    return results_df


# ======= Initialize OpenAI Biomedical LLM Instance =======
@st.cache_resource
def load_openai_llm():
    """
    Load and return an instance of OpenAI's ChatOpenAI LLM.
    
    Returns:
        ChatOpenAI: The initialized LLM instance.
    """
    return ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.1,
        max_tokens=1024
    )

def get_explanation_sections(
        disease1_name, overlapping_genes1,
        disease2_name, overlapping_genes2, llm):
    """
    Generates detailed, rubric-focused explanation sections
    (≈150 words or 6-8 bullets each) for the two diseases,
    their relationship, and the overlapping genes.
    """
    bullet_hint = "• "  # use a real bullet so ChatGPT tends to respond in list form

    # Compute the union of overlapping genes as a set
    genes_overlap = set(overlapping_genes1).intersection(overlapping_genes2)
    genes_overlap_list = sorted(genes_overlap)
    genes_overlap_str = ", ".join(genes_overlap_list) if genes_overlap_list else 'None'

    sections = [
        # 1 ────────────── Disease 1 ──────────────
        (f"Key Features and Genetic Associations for {disease1_name}",
 f"""
You are an expert biomedical scientific editor writing for NIH study-section reviewers.
In ≈150 words **or** 6-8 {bullet_hint}bullets, present a mini-review of **{disease1_name}** that covers:

{bullet_hint}Epidemiology (incidence/prevalence, age/sex bias).  
{bullet_hint}Core clinical phenotype & diagnostic hallmarks.  
{bullet_hint}Major pathophysiological pathways (cell types, signaling cascades).  
{bullet_hint}≥3 high-impact genes/loci **with rsID or OMIM ID**; include effect direction & mechanism.  
{bullet_hint}Notable gene–environment or polygenic-risk insights.  
{bullet_hint}Key therapeutic targets emerging from genetics.
"""),

        # 2 ────────────── Disease 2 ──────────────
        (f"Key Features and Genetic Associations for {disease2_name}",
 f"""
You are an expert biomedical scientific editor writing for NIH study-section reviewers.
In ≈150 words **or** 6-8 {bullet_hint}bullets, present a mini-review of **{disease2_name}** that covers:

{bullet_hint}Epidemiology (incidence/prevalence, age/sex bias).  
{bullet_hint}Core clinical phenotype & diagnostic hallmarks.  
{bullet_hint}Major pathophysiological pathways (cell types, signaling cascades).  
{bullet_hint}≥3 high-impact genes/loci **with rsID or OMIM ID**; include effect direction & mechanism.  
{bullet_hint}Notable gene–environment or polygenic-risk insights.  
{bullet_hint}Key therapeutic targets emerging from genetics.
"""),

        # 3 ────────────── Relationship ──────────────
        (f"Relationship Between {disease1_name} and {disease2_name}",
 f"""
You are an expert biomedical scientific editor.
In ≈120 words discuss **why and how {disease1_name} and {disease2_name} intersect**, covering:

{bullet_hint}Shared molecular or inflammatory pathways.  
{bullet_hint}Clinical or epidemiological comorbidity evidence.  
{bullet_hint}How the overlapping genes ({genes_overlap_str}) mechanistically link the diseases.  
{bullet_hint}Implications for risk stratification or therapy.
"""),

        # 4 ────────────── Overlapping genes ──────────────
        ("Role of Overlapping Genes",
 f"""
You are an expert biomedical geneticist.
Return **one bullet per gene** in this exact markdown format:  
`[GENE]: ↑↓ concise (1-2 sentence) explanation including key variant(s), pathway, and how it drives BOTH diseases`.

Genes to cover: {genes_overlap_str}.
Do **not** repeat bullets or combine genes.
""")
    ]

    # ---------- parallel LLM calls ----------
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            llm.predict_messages,
            [
                {"role": "system", "content": "You are an expert biomedical scientific editor."},
                {"role": "user", "content": prompt}
            ])
            for _, prompt in sections]
        for (header, _), future in zip(sections, futures):
            try:
                results[header] = future.result().content.strip()
            except Exception as e:
                results[header] = f"Error: {e}"
    return results

# ======= LLM Summary Function =======
def get_disease_summary(mesh_code, llm):
    """
    Generates a short summary for a given MeSH code.
    
    Args:
        mesh_code (str): The MeSH code.
        llm (ChatOpenAI): The LLM instance.
    
    Returns:
        str: The generated summary text.
    """
    prompt = f"""You are an expert in biomedical informatics. Provide a short summary (2-3 sentences) for the disease group 
represented by MeSH code "{mesh_code}", including its major clinical features and any known genetic associations."""
    messages = [
        {"role": "system", "content": "You are an expert in biomedical informatics."},
        {"role": "user", "content": prompt}
    ]
    response = llm.predict_messages(messages)
    return response.content.strip()