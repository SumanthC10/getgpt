document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const diseaseQueryInput = document.getElementById('disease-query');
    const topKSelect = document.getElementById('top-k-select');
    const efoResultsList = document.getElementById('efo-results-list');
    const fetchGenesBtn = document.getElementById('fetch-genes-btn');
    const geneTablesContainer = document.getElementById('gene-tables');
    const combineGenesBtn = document.getElementById('combine-genes-btn');

    searchBtn.addEventListener('click', async () => {
        const query = diseaseQueryInput.value.trim();
        const topK = topKSelect.value;
        if (!query) return;

        const response = await fetch(`/v1/efo_search?q=${query}&top_k=${topK}`);
        const data = await response.json();
        
        efoResultsList.innerHTML = '';
        data.results.forEach(item => {
            const li = document.createElement('li');
            li.textContent = `${item.label} (${item.efo_id}) - Score: ${item.score.toFixed(2)}`;
            li.dataset.efoId = item.efo_id;
            li.addEventListener('click', () => {
                li.classList.toggle('selected');
                const selectedCount = efoResultsList.querySelectorAll('.selected').length;
                fetchGenesBtn.style.display = selectedCount > 0 ? 'block' : 'none';
            });
            efoResultsList.appendChild(li);
        });
    });

    fetchGenesBtn.addEventListener('click', async () => {
        const selectedItems = efoResultsList.querySelectorAll('.selected');
        if (selectedItems.length === 0) return;

        geneTablesContainer.innerHTML = '';
        for (const item of selectedItems) {
            const efoId = item.dataset.efoId;
            const response = await fetch('/v1/get-list', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disease_id: efoId })
            });
            const data = await response.json();
            createGeneTable(efoId, data.results);
        }
        combineGenesBtn.style.display = 'block';
    });

    combineGenesBtn.addEventListener('click', () => {
        const allTables = geneTablesContainer.querySelectorAll('table');
        const combinedData = {};

        allTables.forEach(table => {
            const efoId = table.dataset.efoId;
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(row => {
                const geneSymbol = row.cells[0].textContent;
                if (!combinedData[geneSymbol]) {
                    combinedData[geneSymbol] = { efoIds: new Set() };
                }
                combinedData[geneSymbol].efoIds.add(efoId);
            });
        });

        createCombinedTable(combinedData);
    });

    function createGeneTable(efoId, genes) {
        const table = document.createElement('table');
        table.dataset.efoId = efoId;
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        
        thead.innerHTML = `<tr><th colspan="2">${efoId}</th></tr><tr><th>Gene Symbol</th><th>Source</th></tr>`;
        
        genes.forEach(gene => {
            const row = tbody.insertRow();
            row.insertCell().textContent = gene.gene_symbol;
            row.insertCell().textContent = gene.source;
        });

        table.append(thead, tbody);
        geneTablesContainer.appendChild(table);
    }

    function createCombinedTable(combinedData) {
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');

        thead.innerHTML = '<tr><th>Gene Symbol</th><th>Associated EFO IDs</th></tr>';

        for (const geneSymbol in combinedData) {
            const row = tbody.insertRow();
            row.insertCell().textContent = geneSymbol;
            row.insertCell().textContent = Array.from(combinedData[geneSymbol].efoIds).join(', ');
        }

        table.append(thead, tbody);
        geneTablesContainer.innerHTML = ''; // Clear existing tables
        geneTablesContainer.appendChild(table);
        combineGenesBtn.style.display = 'none';
    }
});