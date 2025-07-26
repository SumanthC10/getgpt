from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Configuration
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DATA_FILES = [
    'data/d2g_final_new.csv',
    'data/DGIdb_2_3_25.csv',
    'data/efo_embeddings.parquet'
]

def get_last_updated_date():
    """Gets the most recent modification date from the data files."""
    latest_timestamp = 0
    for file_path in DATA_FILES:
        try:
            timestamp = os.path.getmtime(file_path)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
        except FileNotFoundError:
            # Handle case where a file might be missing
            pass
    if latest_timestamp:
        return datetime.fromtimestamp(latest_timestamp).strftime('%b %d, %Y')
    return "N/A"


@app.route('/')
def index():
    last_updated = get_last_updated_date()
    return render_template('index.html', api_url=API_BASE_URL, last_updated=last_updated)

@app.route('/docs')
def docs():
    return render_template('docs.html', api_url=API_BASE_URL)

@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html', api_url=API_BASE_URL)

@app.route('/analyze')
def analyze():
    return render_template('analyze.html', api_url=API_BASE_URL)
@app.route('/v1/efo_search', methods=['GET'])
def efo_search():
    query = request.args.get('q')
    top_k = request.args.get('top_k')
    response = requests.get(f"{API_BASE_URL}/v1/efo_search", params={"q": query, "top_k": top_k})
    return jsonify(response.json())

@app.route('/v1/get-list', methods=['POST'])
def get_list():
    response = requests.post(f"{API_BASE_URL}/v1/get-list", json=request.json)
    return jsonify(response.json())

@app.route('/v1/genes/pager', methods=['POST'])
def pager_analysis():
    """Proxies PAGER analysis requests to the backend API."""
    response = requests.post(f"{API_BASE_URL}/v1/genes/pager", json=request.json)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5000, debug=True)