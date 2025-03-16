import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf_file(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load_and_split()
    full_text = "\n".join(page.page_content for page in pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    texts = text_splitter.split_text(full_text)
    return full_text, texts
