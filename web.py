from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Configuration
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

@app.route('/')
def index():
    return render_template('index.html', api_url=API_BASE_URL)

@app.route('/docs')
def docs():
    return render_template('docs.html', api_url=API_BASE_URL)

@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html', api_url=API_BASE_URL)
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

if __name__ == '__main__':
    app.run(port=5000, debug=True)