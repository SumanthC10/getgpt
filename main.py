import streamlit as st
import pandas as pd
import json
import time
import logging
import concurrent.futures

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
from opentargets_api import execute_query, execute_genetics_query
from pubmed import query_pubmed_for_abstracts
from gene_extraction import extract_gene_names, extract_genes_with_chatgpt
from pdf_processing import read_pdf_file
from semantic_search import load_mesh_data, search_mesh, precompute_embeddings
from disgenet_utils import ds_get_disgenet_geneset_to_disease, ds_select_best_umls_term

# Logging configuration
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ChatOpenAI for gene extraction (if needed in other parts)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.1,
    max_tokens=4096
)

def test_opentargets_api():
    try:
        test_query = """
        {
          meta {
            apiVersion {
              x
              y
              z
            }
          }
        }
        """
        response = execute_query(test_query)
        if 'errors' in response:
            logger.error(f"GraphQL errors: {json.dumps(response['errors'], indent=2)}")
            return f"Error querying OpenTargets Platform API: {response['errors'][0]['message']}"
        version = response["data"]["meta"]["apiVersion"]
        platform_message = f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}"
        test_genetics_query = """
        query {
          meta {
            apiVersion {
              major
              minor
              patch
            }
          }
        }
        """
        genetics_response = execute_genetics_query(test_genetics_query)
        if 'errors' in genetics_response:
            logger.error(f"GraphQL errors in Genetics API: {json.dumps(genetics_response['errors'], indent=2)}")
            return f"Error querying OpenTargets Genetics API: {genetics_response['errors'][0]['message']}"
        genetics_version = genetics_response["data"]["meta"]["apiVersion"]
        genetics_message = f"Successfully connected to OpenTargets Genetics API. Version: {genetics_version['major']}.{genetics_version['minor']}.{genetics_version['patch']}"
        return f"{platform_message}<br>{genetics_message}"
    except Exception as e:
        logger.exception("Error testing OpenTargets APIs")
        return f"Error testing OpenTargets APIs: {str(e)}"

def query_opentargets(disease_name):
    try:
        # First query: Search for the disease by MeSH term.
        search_query = """
        query($diseaseText: String!) {
          search(queryString: $diseaseText, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits {
              id
              name
            }
          }
        }
        """
        search_data = execute_query(search_query, {"diseaseText": disease_name})
        if not search_data.get("data", {}).get("search", {}).get("hits"):
            return f"No disease found for '{disease_name}'", None
        disease_id = search_data["data"]["search"]["hits"][0]["id"]
        disease_name_found = search_data["data"]["search"]["hits"][0]["name"]
        logger.info(f"Disease search successful. ID: {disease_id}, Name: {disease_name_found}")
        result = f"Information for {disease_name_found} (ID: {disease_id}):\n\n"
        
        # Query OpenTargets Genetics API.
        variants_query = """
        query StudyVariants($studyId: String!) {
          manhattan(studyId: $studyId) {
            associations {
              variant {
                id
                rsId
              }
              pval
              bestGenes {
                score
                gene {
                  id
                  symbol
                }
              }
            }
          }
        }
        """
        study_id = "GCST90002369"  # Update as needed.
        variants_data = execute_genetics_query(variants_query, {"studyId": study_id})
        genetics_genes = {}
        if 'errors' in variants_data:
            logger.error(f"GraphQL errors in variants query: {json.dumps(variants_data['errors'], indent=2)}")
            result += f"\nError fetching variants data: {variants_data['errors'][0]['message']}\n"
        else:
            result += "OpenTargets Genetics Results:\n"
            associations = variants_data["data"]["manhattan"]["associations"]
            for association in associations[:10]:
                variant = association["variant"]
                result += f"Variant: {variant['id']} (rsID: {variant['rsId']})\n"
                result += f"p-value: {association['pval']}\n"
                result += "Best Genes:\n"
                for best_gene in association["bestGenes"]:
                    gene = best_gene["gene"]
                    genetics_genes[gene['symbol']] = best_gene['score']
                    result += f"  - {gene['symbol']} (ID: {gene['id']}, Score: {best_gene['score']})\n"
                result += "\n"
        
        # Query PubMed and extract genes using ChatGPT.
        abstracts = query_pubmed_for_abstracts(disease_name_found)
        chatgpt_genes_text = extract_genes_with_chatgpt(abstracts, "PubMed abstracts", llm)
        result += "\nPubMed and ChatGPT Results:\n"
        result += "Differentially expressed genes extracted from PubMed abstracts:\n"
        result += chatgpt_genes_text + "\n"
        chatgpt_genes = set(extract_gene_names(chatgpt_genes_text))
        
        # Query OpenTargets disease details.
        disease_query = """
        query($diseaseId: String!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: {index: 0, size: 100}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """
        disease_data = execute_query(disease_query, {"diseaseId": disease_id})
        opentargets_genes = {}
        if 'errors' in disease_data:
            logger.error(f"GraphQL errors in disease query: {json.dumps(disease_data['errors'], indent=2)}")
            result += f"\nError fetching disease data: {disease_data['errors'][0]['message']}\n"
        else:
            disease_info = disease_data["data"]["disease"]
            result += "\nOpenTargets Results:\n"
            result += f"Total associated targets: {disease_info['associatedTargets']['count']}\n"
            result += "Top 100 associated targets from OpenTargets:\n"
            for i, row in enumerate(disease_info['associatedTargets']['rows'], 1):
                target = row['target']
                opentargets_genes[target['approvedSymbol']] = row['score']
                result += f"{i}. {target['approvedSymbol']} ({target['approvedName']})\n"
                result += f"   ID: {target['id']}, Association Score: {row['score']:.4f}\n"
        
        # Combine genes from all sources.
        all_genes = set(opentargets_genes.keys()) | set(genetics_genes.keys()) | chatgpt_genes
        result += "\nUnique Gene Analysis:\n"
        result += f"Total unique genes found: {len(all_genes)}\n\n"
        genetics_unique = set(genetics_genes.keys()) - set(opentargets_genes.keys()) - chatgpt_genes
        chatgpt_unique = chatgpt_genes - set(opentargets_genes.keys()) - set(genetics_genes.keys())
        opentargets_unique = set(opentargets_genes.keys()) - set(genetics_genes.keys()) - chatgpt_genes
        
        result += f"Genes unique to OpenTargets Genetics ({len(genetics_unique)}):\n"
        for gene in sorted(genetics_unique):
            result += f"- {gene} (Score: {genetics_genes[gene]:.4f})\n"
        result += "\n"
        result += f"Genes unique to ChatGPT analysis ({len(chatgpt_unique)}):\n"
        for gene in sorted(chatgpt_unique):
            result += f"- {gene}\n"
        result += "\n"
        result += f"Genes unique to OpenTargets ({len(opentargets_unique)}):\n"
        for gene in sorted(opentargets_unique):
            result += f"- {gene} (Score: {opentargets_genes[gene]:.4f})\n"
        result += "\n"
        
        gene_data = []
        for gene in sorted(all_genes):
            gene_data.append({
                "Gene": gene,
                "OpenTargets Score": opentargets_genes.get(gene, "N/A"),
                "Genetics API Score": genetics_genes.get(gene, "N/A"),
                "Found in PubMed": "Yes" if gene in chatgpt_genes else "No"
            })
        
        return result, gene_data

    except Exception as e:
        logger.exception(f"Error in query_opentargets: {str(e)}")
        return f"An error occurred while querying OpenTargets and extracting genes: {str(e)}", None

def toggle_run():
    st.session_state["is_running"] = not st.session_state.get("is_running", False)
    if not st.session_state["is_running"]:
        st.session_state["stop_query"] = True

# -----------------------------
# MAIN AUTOMATIC G2D + EXPLANATION FLOW
# -----------------------------
def main():
    st.title("Disease-To-Geneset and Gene Analysis")
    
    # Initialize session state variables if not already set.
    for key in ["stop_query", "is_running", "gene_list", "gene_table", "results_df", "base_disease", "selected_mesh_d2g", "selected_mesh_d2g_code"]:
        if key not in st.session_state:
            st.session_state[key] = None if key in ["gene_table", "results_df", "base_disease", "selected_mesh_d2g", "selected_mesh_d2g_code"] else False
    if st.session_state["gene_list"] is None:
        st.session_state["gene_list"] = []

    # Load MeSH data from semantic_search functions.
    mesh_preferred_terms, mesh_alt_terms, preferred_embeddings, alt_embeddings = load_mesh_data()
    mesh_mapping = load_mesh_mapping()

    tabs = st.tabs(["Disease-To-Geneset (D2G) Associations", "API Testing"])
    
    # --- First Tab: D2G Associations ---
    with tabs[0]:
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
            # If a new disease is selected, reset analysis state.
            if st.session_state.get("selected_mesh_d2g") != selected_mesh_d2g:
                st.session_state["gene_list"] = []
                st.session_state["gene_table"] = None
                st.session_state["results_df"] = None
                st.session_state["base_disease"] = selected_mesh_d2g
            st.session_state["selected_mesh_d2g"] = selected_mesh_d2g

        pdf_files = st.file_uploader("Drag and drop your PDF file for gene extraction here", 
                                     type="pdf", accept_multiple_files=True)
        
        toggle_label = "Stop Query" if st.session_state["is_running"] else "Run Analysis"
        st.button(toggle_label, on_click=toggle_run, key="toggle_button")
        
        # Run the query if "is_running" is True.
        if st.session_state["is_running"]:
            if not selected_mesh_d2g or selected_mesh_d2g not in mesh_preferred_terms:
                st.error("Please select a valid MeSH term from the suggestions before running the analysis.")
            else:
                st.session_state["base_disease"] = selected_mesh_d2g
                disease_name = selected_mesh_d2g
                with st.spinner("Querying OpenTargets, Genetics API, and PubMed abstracts..."):
                    result_text, gene_data = query_opentargets(disease_name)
                if gene_data:
                    df_genes = pd.DataFrame(gene_data)
                    # Reset the index to start at 1
                    df_genes.index = range(1, len(df_genes) + 1)
                    st.session_state["gene_list"] = sorted(set(df_genes["Gene"].tolist()))
                    st.session_state["gene_table"] = df_genes.copy()
                else:
                    df_genes = pd.DataFrame(columns=["Gene", "OpenTargets Score", "Genetics API Score", "Found in PubMed"])
                    st.session_state["gene_table"] = df_genes.copy()
                
                if pdf_files is not None:
                    pdf_genes_set = set()
                    for file_index, pdf_file in enumerate(pdf_files, start=1):
                        with st.spinner(f"Processing PDF file {file_index} and extracting genes..."):
                            full_text, texts = read_pdf_file(pdf_file)
                            gene_extraction_responses = []
                            total_chunks = len(texts)
                            for chunk_index, chunk in enumerate(texts, start=1):
                                if st.session_state.get("stop_query", False):
                                    st.error("Query Stopped")
                                    break
                                with st.spinner(f"Extracting genes from PDF file {file_index} - chunk {chunk_index}/{total_chunks}..."):
                                    response = extract_genes_with_chatgpt(chunk, f"PDF file {file_index} - chunk {chunk_index}", llm)
                                    gene_extraction_responses.append(response)
                            if st.session_state.get("stop_query", False):
                                break
                            combined_gene_extraction = "\n".join(gene_extraction_responses)
                            extracted_genes = extract_gene_names(combined_gene_extraction)
                            pdf_genes_set = pdf_genes_set.union(set(extracted_genes))
                    
                    if pdf_genes_set:
                        df_genes = st.session_state["gene_table"]
                        found_in_pdf = ["Yes" if gene in pdf_genes_set else "No" for gene in df_genes["Gene"]]
                        df_genes["Found in PDF"] = found_in_pdf
                        missing_genes = pdf_genes_set - set(df_genes["Gene"])
                        if missing_genes:
                            additional_rows = [{
                                "Gene": gene,
                                "OpenTargets Score": "N/A",
                                "Genetics API Score": "N/A",
                                "Found in PubMed": "No",
                                "Found in PDF": "Yes"
                            } for gene in missing_genes]
                            df_additional = pd.DataFrame(additional_rows)
                            df_genes = pd.concat([df_genes, df_additional], ignore_index=True)
                            # Reset the index again to start at 1 after concatenation
                            df_genes.index = range(1, len(df_genes) + 1)
                        st.session_state["gene_table"] = df_genes.copy()
                
                # Compute overlap analysis immediately after gene extraction.
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
                            )
                            overlap_results = overlap_results.rename(columns={"MeSH_code": "MeSH_Code"})
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
                
                st.session_state["is_running"] = False  # Finished query and overlap analysis.
        
        # Display results if available.
        if st.session_state.get("gene_table") is not None and st.session_state.get("gene_list"):
            disease_filename = (
                st.session_state["selected_mesh_d2g"].replace(" ", "_")
                if st.session_state.get("selected_mesh_d2g") else "analysis"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Geneset")
                st.dataframe(st.session_state["gene_table"])
                geneset_csv = st.session_state["gene_table"].to_csv(index=False)
                st.download_button(
                    label="Download Geneset CSV",
                    data=geneset_csv,
                    file_name=f"{disease_filename}_gene_analysis.csv",
                    mime="text/csv",
                )
            with col2:
                if st.session_state.get("results_df") is not None:
                    st.subheader("Top Overlapping Diseases")
                    cols = [col for col in ["Disease_Name", "Overlap_Count", "p_value", "MeSH_Code"] if col in st.session_state["results_df"].columns]
                    st.write(
                        st.session_state["results_df"][cols].style.format({"p_value": "{:.2e}"})
                    )
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
            descriptors_df = load_mesh_descriptors()  # Ensure descriptors are loaded.
            disease_list = st.session_state["results_df"]["Disease_Name"].tolist()
            selected_disease2 = st.selectbox("Select Disease for Explanation", disease_list, key="explanation_disease2")
            if st.button("Get Explanation", key="get_explanation"):
                spinner_placeholder = st.empty()
                spinner_placeholder.info("Getting explanation, please wait...")
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

                    llm_instance = load_openai_llm()
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
    
    # --- Second Tab: API Testing ---
    with tabs[1]:
        st.header("API Test")
        if st.button("Test OpenTargets APIs"):
            with st.spinner("Testing APIs..."):
                result = test_opentargets_api()
            st.markdown(result, unsafe_allow_html=True)

if __name__ == "__main__":
    main()