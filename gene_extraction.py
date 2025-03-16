import re
import logging
import streamlit as st

logger = logging.getLogger(__name__)

def extract_gene_names(text):
    gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
    return re.findall(gene_pattern, text)

def extract_genes_with_chatgpt(text, context_label, llm):
    try:
        prompt = f"""As an expert in genomics and bioinformatics, extract a list of the differentially expressed genes from the following {context_label} text.
Output the gene symbols in a numbered list.

Text:
{text}

Genes:"""
        messages = [
            {"role": "system", "content": "You are an expert in genomics and bioinformatics."},
            {"role": "user", "content": prompt}
        ]
        response = llm.predict_messages(messages)
        return response.content.strip()
    except Exception as e:
        logger.exception(f"Error in ChatGPT gene extraction: {str(e)}")
        return f"An error occurred while extracting genes with ChatGPT: {str(e)}"
