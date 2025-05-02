# GetGPT

## Overview

GetGPT is a Streamlit-based application for exploring disease–gene relationships. It integrates data from OpenTargets, PubMed, MeSH, and leverages OpenAI's GPT-4.1 for summarization to generate PAG (Published Association Graph) reports.

## Features

- **OpenTargets API**: Retrieve associated targets and genetic variants.
- **PubMed Integration**: Query abstracts and extract differentially expressed genes.
- **MeSH & EFO/MONDO Conversion**: Map disease names to standard IDs.
- **Semantic Search**: Precompute embeddings and search MeSH descriptors.
- **PAG Backend**: Analyze and package gene association graphs into downloadable reports.
- **PDF Processing**: Read and process PDF documents.
- **LLM Summaries**: Use GPT-4.1 to generate human-readable disease summaries.
- **Interactive Streamlit UI**: Explore data, run analyses, and download results.

## Installation

### Prerequisites

- Python 3.13+ or Anaconda/Miniconda
- ChromeDriver (for Selenium)
- OpenAI API Key

### Setup with Conda

```bash
conda env create -f environment.yml
conda activate getgpt
```

### Configure OpenAI Key

Create a file at `~/.streamlit/secrets.toml` with:

```toml
[general]
OPENAI_API_KEY = "your_api_key_here"
```

Ensure `chromedriver` is installed and on your `PATH`:

```bash
brew install chromedriver
```

## Usage

Launch the Streamlit app:

```bash
streamlit run main.py
```

Access the app in your browser at `http://localhost:8501`.

## Project Structure

```
├── main.py               # Streamlit app entrypoint
├── g2d_utils.py          # Utilities for G2D data and LLM integration
├── opentargets_api.py    # Wrappers for OpenTargets API
├── pubmed.py             # PubMed querying and abstract parsing
├── gene_extraction.py    # Gene name extraction logic
├── semantic_search.py    # MeSH data loading and semantic search
├── pdf_processing.py     # PDF reading and processing
├── pager_backend.py      # Core PAG analysis and report generation
├── efo_conversion.py     # EFO/MONDO ID conversion utilities
├── autocomplete_api.py   # Autocomplete API utilities
├── mesh_umls_subset.py   # Scripts for MeSH/UMLS subset processing
├── data/                 # Directory for cached datasets
├── setup/                # Setup scripts for MeSH/UMLS data
│   ├── mesh_umls.sh
│   └── subset_mesh.py
├── environment.yml       # Conda environment specification
└── README.md             # Project documentation
```

## Configuration

- **OPENAI_API_KEY**: Set via Streamlit secrets or environment variable.
- **ChromeDriver**: Required for Selenium-based scraping; ensure it matches your Chrome version.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
