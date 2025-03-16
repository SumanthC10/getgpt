import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import logging
import ast
from langchain_openai import ChatOpenAI
import networkx.algorithms.community as nx_comm

# Set up logging.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------
# Part 1: Network Functions (Gene-Disease)
# ------------------------
def build_disease_gene_mapping(df, disease_col="Disease_Name", gene_col="Overlapping_Genes"):
    """
    Build a dictionary mapping each disease (from the overlap file)
    to the set of genes.
    If gene_col is "Overlapping_Genes", each cell is parsed from its string representation.
    """
    mapping = {}
    for _, row in df.iterrows():
        disease = row[disease_col]
        # For the new file structure, parse the string representation of the gene list.
        if gene_col == "Overlapping_Genes":
            try:
                genes = ast.literal_eval(row[gene_col])
            except Exception as e:
                logger.error(f"Error parsing genes for disease {disease}: {e}")
                genes = []
        else:
            genes = [row[gene_col]]
        mapping.setdefault(disease, set()).update(genes)
    return mapping

def build_centered_network(filtered_df, central_disease, central_genes,
                           disease_col="Disease_Name", gene_col="Overlapping_Genes"):
    """
    Build a network where:
      - The central disease is forced to be the hub.
      - Only diseases that share at least one gene with the central disease are included.
      - An edge is added between the central disease and every other disease (with the shared genes as attribute).
      - Additionally, edges are added among non-central diseases if they share any genes.
    Returns:
      - G: the networkx graph
      - mapping: dictionary mapping each disease to its set of genes (from filtered_df)
    """
    mapping = build_disease_gene_mapping(filtered_df, disease_col, gene_col)

    # -- FIX: Ensure the base disease has the uploaded geneset --
    if central_disease not in mapping:
        mapping[central_disease] = set(central_genes)
    else:
        mapping[central_disease].update(central_genes)
    # -----------------------------------------------------------

    # Collect all diseases that share at least one of the central genes
    diseases_to_include = {
        d for d, genes in mapping.items()
        if d == central_disease or genes.intersection(central_genes)
    }

    G = nx.Graph()
    for disease in diseases_to_include:
        G.add_node(disease)

    # Edges from central disease to others
    for disease in diseases_to_include:
        if disease == central_disease:
            continue
        common_with_central = mapping[disease].intersection(central_genes)
        if common_with_central:
            G.add_edge(central_disease, disease, overlap=list(common_with_central))

    # Edges among non-central diseases if they share any genes
    diseases_list = list(diseases_to_include)
    for i in range(len(diseases_list)):
        for j in range(i + 1, len(diseases_list)):
            d1, d2 = diseases_list[i], diseases_list[j]
            if d1 == central_disease or d2 == central_disease:
                continue
            common = mapping[d1].intersection(mapping[d2])
            if common:
                G.add_edge(d1, d2, overlap=list(common))

    return G, mapping

def get_edge_color(count, min_count, max_count):
    """
    Use a colormap to generate a color based on the count of shared genes.
    """
    norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
    cmap = cm.get_cmap('viridis')
    rgb = cmap(norm(count))[:3]
    return mcolors.to_hex(rgb)

def plot_centered_network_interactive(G, central_disease, title="Centered Disease Network"):
    """
    Create an interactive network visualization using pyvis.
    Returns the HTML string for the visualization.
    """
    net = Network(height="600px", width="100%", notebook=True, heading=title)

    for node in G.nodes():
        if node == central_disease:
            net.add_node(node,
                         label=node,
                         shape="box",
                         color={"background": "red", "border": "black",
                                "highlight": {"background": "red", "border": "black"}},
                         font={"color": "white", "size": 20})
        else:
            net.add_node(node,
                         label=node,
                         shape="box",
                         color={"background": "#f0f0f0", "border": "gray",
                                "highlight": {"background": "#d3d3d3", "border": "gray"}},
                         font={"color": "black", "size": 20})

    edge_counts = [len(data.get("overlap", [])) for _, _, data in G.edges(data=True)]
    if edge_counts:
        min_count, max_count = min(edge_counts), max(edge_counts)
    else:
        min_count, max_count = 1, 1

    for source, target, data in G.edges(data=True):
        overlap = data.get("overlap", [])
        hover_text = "Genes: " + ", ".join(overlap)
        count = len(overlap)
        width = count * 2
        edge_color = get_edge_color(count, min_count, max_count)
        net.add_edge(source, target, title=hover_text, width=width, color=edge_color)

    net.set_options("""
    var options = {
      "layout": {"improvedLayout": true},
      "nodes": {"font": {"size": 20, "face": "Tahoma", "align": "center"}},
      "edges": {"smooth": false},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 50,
          "springConstant": 0,
          "avoidOverlap": 0.05
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      }
    }
    """)
    return net.generate_html()

# ------------------------
# Part 2: Drug-to-Gene and Drug Repurposing Functions (Simple Mapping)
# ------------------------
def build_gene_drug_mapping(drug_df):
    """
    Build a dictionary mapping each gene to a set of drugs.
    Expects the drug CSV to have columns: gene, interaction_type, score, drug_name.
    This is a simple mapping (without detailed info).
    """
    mapping = {}
    for _, row in drug_df.iterrows():
        gene = row["gene"]
        drug = row["drug_name"]
        mapping.setdefault(gene, set()).add(drug)
    return mapping

def compute_disease_drug_candidates(disease_mapping, gene_drug_mapping):
    """
    For each disease (key in disease_mapping), compute candidate drugs
    by taking the union of drugs for each gene in its gene set.
    Returns a DataFrame with columns: Disease, Candidate_Drugs.
    """
    records = []
    for disease, genes in disease_mapping.items():
        candidate_drugs = set()
        for gene in genes:
            candidate_drugs.update(gene_drug_mapping.get(gene, set()))
        records.append({
            "Disease": disease,
            "Candidate_Drugs": ", ".join(sorted(candidate_drugs))
        })
    return pd.DataFrame(records)

# ------------------------
# Part 3: Main App (Basic Workflow)
# ------------------------
def main():
    st.title("Cluster Diseases & Drug Repurposing (Interactive)")

    st.markdown("""
    **Workflow:**
    1. Upload two CSV files:
       - **Gene–Disease Overlap CSV:** Contains gene–disease associations.
         (Expected columns: **MeSH_Code**, **Total_Aggregated_Genes**, **Overlap_Count**, **p_value**, **Overlapping_Genes**, **Disease_Name**)
       - **Geneset CSV:** Contains genes associated with your initial disease (expected column: **Gene** or **Gene Symbol**).
    2. Enter the **Initial Disease Name** (this is the central disease).
    3. The app will:
         - Use the genes from the Geneset CSV as the central disease’s genes.
         - Filter the Gene–Disease Overlap CSV based on these genes.
         - Build and display a network where the initial disease is the hub.
         - Read the Drug–Gene file (DGIdb_2_3_25.csv) from disk and compute candidate drugs for each disease (using a simple mapping).
         - Display a table of candidate drugs for repurposing across diseases.
    """)

    st.sidebar.header("Upload Files and Disease Input")
    overlap_file = st.sidebar.file_uploader("Upload Gene–Disease Overlap CSV", type=["csv"], key="overlap")
    geneset_file = st.sidebar.file_uploader("Upload Geneset CSV", type=["csv"], key="geneset")
    initial_disease = st.sidebar.text_input("Enter Initial Disease Name", value="")

    if overlap_file is not None and geneset_file is not None and initial_disease:
        try:
            # Load the overlap and geneset data.
            overlap_df = pd.read_csv(overlap_file)
            st.subheader("Gene–Disease Overlap Data (first few rows)")
            st.dataframe(overlap_df.head())

            geneset_df = pd.read_csv(geneset_file)
            st.subheader("Geneset Data (first few rows)")
            st.dataframe(geneset_df.head())

            # Detect columns for the new file structure.
            overlap_columns = overlap_df.columns
            if "Disease_Name" in overlap_columns:
                disease_col = "Disease_Name"
            else:
                st.error("Overlap CSV must contain a disease column (expected: 'Disease_Name').")
                return

            if "Overlapping_Genes" in overlap_columns:
                gene_col = "Overlapping_Genes"
            else:
                st.error("Overlap CSV must contain a gene column (expected: 'Overlapping_Genes').")
                return

            # Ensure the geneset CSV has the correct column.
            if "Gene" not in geneset_df.columns:
                if "Gene Symbol" in geneset_df.columns:
                    geneset_df = geneset_df.rename(columns={"Gene Symbol": "Gene"})
                else:
                    st.error("Geneset CSV must contain a column named 'Gene' or 'Gene Symbol'.")
                    return

            central_genes = set(geneset_df["Gene"].dropna().unique())
            st.markdown(f"**Initial Disease:** {initial_disease}")
            st.markdown(f"**Genes associated with {initial_disease}:** {sorted(central_genes)}")

            # Filter the overlap data by checking if any overlapping gene is in the central gene set.
            filtered_overlap_df = overlap_df[
                overlap_df[gene_col].apply(
                    lambda x: len(set(ast.literal_eval(x)).intersection(central_genes)) > 0
                )
            ]
            st.subheader("Filtered Gene–Disease Overlap Data")
            st.dataframe(filtered_overlap_df)

            # Build disease network.
            G, disease_mapping = build_centered_network(
                filtered_overlap_df,
                initial_disease,
                central_genes,
                disease_col=disease_col,
                gene_col=gene_col
            )
            st.subheader("Centered Disease Network (Interactive)")
            st.markdown(f"**Diseases in the Network:** {sorted(G.nodes())}")
            html_str = plot_centered_network_interactive(
                G,
                initial_disease,
                title=f"Network Centered on {initial_disease}"
            )
            components.html(html_str, height=600)

            # Load drug-gene data from disk.
            drug_df = pd.read_csv("data/DGIdb_2_3_25.csv")
            st.subheader("Drug–Gene Data (first few rows)")
            st.dataframe(drug_df.head())

            # Simple drug mapping & candidate drugs per disease.
            gene_drug_mapping = build_gene_drug_mapping(drug_df)
            drug_candidates_df = compute_disease_drug_candidates(disease_mapping, gene_drug_mapping)
            st.subheader("Candidate Drugs for Each Disease (Simple Mapping)")
            st.dataframe(drug_candidates_df)

            # ------------------------
            # Part 4: Advanced Drug Repurposing Analysis
            # ------------------------
            st.subheader("Advanced Drug Repurposing Analysis (Cluster-Based)")
            # Build a detailed mapping including interaction type and score.
            def build_gene_drug_mapping_detailed(drug_df):
                """
                Build a dictionary mapping each gene to a list of drug-interaction dictionaries:
                  { gene: [ { "drug_name": str, "interaction_type": str, "score": float }, ... ], ... }
                Expects the drug CSV to have columns: gene, interaction_type, score, drug_name.
                """
                mapping = {}
                for _, row in drug_df.iterrows():
                    gene = row.get("gene")
                    if pd.isna(gene):
                        continue
                    drug_info = {
                        "drug_name": str(row.get("drug_name", "Unknown")),
                        "interaction_type": str(row.get("interaction_type", "Unknown")),
                        "score": float(row.get("score", 0.0))
                    }
                    mapping.setdefault(gene, []).append(drug_info)
                return mapping

            def advanced_cluster_drug_analysis(G, disease_mapping, gene_drug_mapping_detailed):
                """
                Perform community detection on the disease network and analyze candidate drugs for each cluster
                using detailed drug-target information.
                For each community:
                  - Compute the union of genes across all diseases.
                  - Aggregate candidate drugs by counting how many genes they target (synergy),
                    averaging their scores, and collecting interaction types.
                Returns a DataFrame with columns: Cluster_ID, Diseases, Union_Genes, Candidate_Drugs.
                """
                communities = list(nx_comm.greedy_modularity_communities(G))
                cluster_results = []
                for i, community in enumerate(communities, start=1):
                    diseases_in_cluster = sorted(list(community))
                    union_genes = set()
                    for d in diseases_in_cluster:
                        union_genes.update(disease_mapping.get(d, set()))
                    # Aggregate detailed drug info for union of genes
                    drug_info_agg = {}
                    for gene in union_genes:
                        if gene not in gene_drug_mapping_detailed:
                            continue
                        for drug_entry in gene_drug_mapping_detailed[gene]:
                            drug_name = drug_entry["drug_name"]
                            if drug_name not in drug_info_agg:
                                drug_info_agg[drug_name] = {
                                    "genes_hit": set(),
                                    "scores": [],
                                    "interaction_types": set()
                                }
                            drug_info_agg[drug_name]["genes_hit"].add(gene)
                            drug_info_agg[drug_name]["scores"].append(drug_entry["score"])
                            drug_info_agg[drug_name]["interaction_types"].add(drug_entry["interaction_type"])
                    # Prepare a summary for candidate drugs in this cluster
                    candidate_drugs = []
                    for dname, info in drug_info_agg.items():
                        synergy_count = len(info["genes_hit"])
                        avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
                        itypes = ", ".join(sorted(info["interaction_types"]))
                        candidate_drugs.append((dname, synergy_count, avg_score, itypes))
                    candidate_drugs.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    candidate_drugs_str = "; ".join([f"{d} [types: {t}] (hits: {s}, avg_score: {a:.2f})" 
                                                     for d, s, a, t in candidate_drugs])
                    cluster_results.append({
                        "Cluster_ID": i,
                        "Diseases": ", ".join(diseases_in_cluster),
                        "Union_Genes": ", ".join(sorted(union_genes)),
                        "Candidate_Drugs": candidate_drugs_str
                    })
                return pd.DataFrame(cluster_results)

            # Build detailed gene-drug mapping and run advanced analysis.
            detailed_mapping = build_gene_drug_mapping_detailed(drug_df)
            advanced_df = advanced_cluster_drug_analysis(G, disease_mapping, detailed_mapping)
            st.dataframe(advanced_df)

            # ------------------------
            # Part 5: LLM Explanations for Disease-Drug-Pathway Mechanisms
            # ------------------------
            st.subheader("LLM Explanations for Disease, Drug, and Pathway Mechanisms")
            # Create a ChatOpenAI instance.
            llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.2, max_tokens=300)

            def get_disease_drug_pathway_explanations(advanced_df, llm):
                """
                For each disease cluster in advanced_df, request an explanation from the LLM
                that covers potential mechanistic pathways, drug-target interactions,
                and clinical relevance.
                Returns a dictionary mapping Cluster_ID to the LLM explanation.
                """
                explanations = {}
                for index, row in advanced_df.iterrows():
                    cluster_id = row["Cluster_ID"]
                    diseases = row["Diseases"]
                    union_genes = row["Union_Genes"]
                    candidate_drugs = row["Candidate_Drugs"]
                    prompt = (
                        f"Explain the potential mechanistic pathways and functional relevance for the following disease cluster. "
                        f"Include details about the drug-target interactions, key signaling pathways, and clinical implications.\n\n"
                        f"Diseases in cluster: {diseases}\n"
                        f"Union of genes: {union_genes}\n"
                        f"Candidate drugs: {candidate_drugs}\n"
                        f"Provide a concise summary in 3-4 sentences."
                    )
                    response = llm(prompt)
                    explanations[cluster_id] = response
                return explanations

            explanations = get_disease_drug_pathway_explanations(advanced_df, llm)
            st.markdown("### LLM Explanations for Each Cluster")
            for cluster_id, explanation in explanations.items():
                st.markdown(f"#### Cluster {cluster_id}")
                st.write(explanation)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Sidebar: LLM Explanation for individual disease (if available)
    if st.session_state.get("results_df") is not None and not st.session_state["results_df"].empty:
        with st.sidebar:
            st.subheader("LLM Explanation")
            descriptors_df = pd.read_csv("mesh_terms.csv")  # Load descriptors from disk.
            disease_list = st.session_state["results_df"]["Disease_Name"].tolist()
            selected_disease2 = st.selectbox("Select Disease for Explanation", disease_list, key="explanation_disease2")
            if st.button("Get Explanation", key="get_explanation"):
                spinner_placeholder = st.empty()
                spinner_placeholder.info("Getting explanation from LLM, please wait...")
                try:
                    base_disease_code = st.session_state["base_disease"]
                    base_row = descriptors_df[descriptors_df["DescriptorUI"] == base_disease_code]
                    if not base_row.empty:
                        base_disease_name = base_row.iloc[0]["PreferredTerm"]
                    else:
                        base_disease_name = str(base_disease_code)
                    overlapping_genes1 = st.session_state["gene_list"]
                    row2 = st.session_state["results_df"][st.session_state["results_df"]["Disease_Name"] == selected_disease2].iloc[0]
                    overlapping_genes2 = row2["Overlapping_Genes"]

                    from g2d_utils import get_explanation_sections  # Ensure we use the updated function.
                    llm_instance = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.1, max_tokens=512)
                    # Request concise explanations (2-3 sentences per section).
                    sections = get_explanation_sections(
                        base_disease_name, overlapping_genes1,
                        selected_disease2, overlapping_genes2,
                        llm_instance
                    )
                    st.session_state["explanation_sections"] = sections
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

if __name__ == "__main__":
    main()