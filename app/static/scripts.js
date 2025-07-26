document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const searchBtn = document.getElementById('search-btn');
    const diseaseQueryInput = document.getElementById('disease-query');
    const topKSelect = document.getElementById('top-k-select');
    const efoResultsList = document.getElementById('efo-results-list');
    const selectedEfosList = document.getElementById('selected-efos-list');
    const getGenesButton = document.getElementById('get-genes-button');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    const geneListContainer = document.getElementById('gene-list-container');
    const loader = document.getElementById('loader');

    // --- State Management ---
    let selectedEfos = JSON.parse(sessionStorage.getItem('selectedEfos')) || [];
    let geneSets = JSON.parse(sessionStorage.getItem('geneSets')) || {};

    const saveState = () => {
        sessionStorage.setItem('selectedEfos', JSON.stringify(selectedEfos));
        sessionStorage.setItem('geneSets', JSON.stringify(geneSets));
    };

    // --- Utility Functions ---
    const showLoader = () => loader.style.display = 'flex';
    const hideLoader = () => loader.style.display = 'none';

    // --- UI Update Functions ---
    const updateSelectedEfosList = () => {
        selectedEfosList.innerHTML = '';
        selectedEfos.forEach(efo => {
            const tag = document.createElement('div');
            tag.className = 'efo-tag';
            tag.textContent = `${efo.label} (${efo.efo_id})`;

            const removeBtn = document.createElement('span');
            removeBtn.className = 'remove-efo';
            removeBtn.textContent = '×';
            removeBtn.onclick = () => removeEfo(efo.efo_id);

            tag.appendChild(removeBtn);
            selectedEfosList.appendChild(tag);
        });
        getGenesButton.style.display = selectedEfos.length > 0 ? 'block' : 'none';
    };

    const removeEfo = (efoId) => {
        selectedEfos = selectedEfos.filter(efo => efo.efo_id !== efoId);
        if (geneSets[efoId]) {
            delete geneSets[efoId];
        }
        updateSelectedEfosList();
        updateGeneList();
        saveState();
    };

    const updateGeneList = () => {
        geneListContainer.innerHTML = '<h2>Gene List</h2>'; // Clear previous tables
        for (const efoId in geneSets) {
            const efo = selectedEfos.find(e => e.efo_id === efoId);
            const label = efo ? efo.label : efoId;
            const table = createGeneTable(efoId, label, geneSets[efoId]);
            geneListContainer.appendChild(table);
        }
        downloadCsvBtn.style.display = Object.keys(geneSets).length > 0 ? 'block' : 'none';
    };

    // --- Event Listeners ---
    searchBtn.addEventListener('click', async () => {
        const query = diseaseQueryInput.value.trim();
        const topK = topKSelect.value;
        if (!query) return;

        showLoader();
        try {
            const response = await fetch(`${window.API_BASE_URL}/v1/efo_search?q=${query}&top_k=${topK}`);
            const data = await response.json();
            efoResultsList.innerHTML = ''; // Clear previous results
            data.results.forEach(item => {
                const li = document.createElement('li');
                li.textContent = `${item.label} (${item.efo_id}) - Score: ${item.score.toFixed(2)}`;
                li.dataset.efoId = item.efo_id;
                li.dataset.label = item.label;
                li.addEventListener('click', () => {
                    if (selectedEfos.length >= 5) {
                        alert("You can select a maximum of 5 EFO IDs.");
                        return;
                    }
                    if (!selectedEfos.some(efo => efo.efo_id === item.efo_id)) {
                        selectedEfos.push({ efo_id: item.efo_id, label: item.label });
                        updateSelectedEfosList();
                        saveState();
                    }
                });
                efoResultsList.appendChild(li);
            });
        } catch (error) {
            console.error("Error fetching EFO results:", error);
        } finally {
            hideLoader();
        }
    });

    diseaseQueryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' || event.keyCode === 13) {
            event.preventDefault();
            searchBtn.click();
        }
    });

    getGenesButton.addEventListener('click', async () => {
        if (selectedEfos.length === 0) return;

        showLoader();
        try {
            for (const efo of selectedEfos) {
                if (geneSets[efo.efo_id]) continue; // Skip if already fetched

                const response = await fetch(`${window.API_BASE_URL}/v1/get-list`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ disease_ids: [efo.efo_id] })
                });
                const data = await response.json();
                geneSets[efo.efo_id] = data.results;
            }
            updateGeneList();
            saveState();
        } catch (error) {
            console.error("Error fetching gene list:", error);
        } finally {
            hideLoader();
        }
    });

    downloadCsvBtn.addEventListener('click', () => {
        let csvContent = "data:text/csv;charset=utf-8,EFO ID,Disease,Gene,Source,G Score,E Score,T Score,Overall Score,Evidence\n";
        for (const efoId in geneSets) {
            const efo = selectedEfos.find(e => e.efo_id === efoId);
            const label = efo ? efo.label : efoId;
            geneSets[efoId].forEach(gene => {
                const row = [
                    efoId,
                    label,
                    gene.gene_symbol,
                    gene.source.map(s => s.name).join(';'),
                    gene.g_score,
                    gene.e_score,
                    gene.t_score,
                    gene.overall_score,
                    formatEvidence(gene.evidence, true)
                ].map(cell => `"${(cell || '').toString().replace(/"/g, '""')}"`).join(",");
                csvContent += row + "\n";
            });
        }

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "gene_sets.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // --- Table Functions ---
    function createGeneTable(efoId, label, genes) {
        const tableContainer = document.createElement('div');
        tableContainer.className = 'gene-table-container';

        const title = document.createElement('h3');
        title.textContent = `${label} (${efoId})`;
        tableContainer.appendChild(title);

        const table = document.createElement('table');
        table.id = `gene-table-${efoId.replace(':', '_')}`;
        table.className = 'display gene-table';
        table.style.width = '100%';

        table.innerHTML = `
            <thead>
                <tr>
                    <th>Gene</th>
                    <th>Source</th>
                    <th>G Score</th>
                    <th>E Score</th>
                    <th>T Score</th>
                    <th>Overall Score</th>
                    <th>Evidence</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        `;
        tableContainer.appendChild(table);

        const tableBody = table.querySelector('tbody');
        genes.forEach(gene => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${gene.gene_symbol}</td>
                <td>${formatSource(gene.source)}</td>
                <td>${gene.g_score ? gene.g_score.toFixed(4) : ''}</td>
                <td>${gene.e_score ? gene.e_score.toFixed(4) : ''}</td>
                <td>${gene.t_score ? gene.t_score.toFixed(4) : ''}</td>
                <td>${gene.overall_score.toFixed(4)}</td>
                <td>${formatEvidence(gene.evidence)}</td>
            `;
            tableBody.appendChild(row);
        });

        // Initialize DataTable after populating
        setTimeout(() => {
            $(`#${table.id}`).DataTable({
                "pageLength": 5,
                "responsive": true,
                "destroy": true
            });
        }, 0);

        return tableContainer;
    }

    function formatEvidence(evidence, forCsv = false) {
        if (!evidence || evidence.length === 0) return '';
        const separator = forCsv ? ';' : ' ';
        if (typeof evidence[0] === 'object' && evidence[0] !== null && 'url' in evidence[0]) {
            return evidence.map(item => forCsv ? `${item.display} (${item.url})` : `<a href="${item.url}" target="_blank" class="evidence-link">${item.display}</a>`).join(separator);
        }
        return JSON.stringify(evidence);
    }

    function formatSource(source) {
        if (!source) return '';
        if (Array.isArray(source)) {
            return source.map(s => `<a href="${s.url}" target="_blank" class="source-link">${s.name}</a>`).join(' ');
        }
        return `<a href="${source.url}" target="_blank" class="source-link">${source.name}</a>`;
    }

    // --- Initialization ---
    const init = () => {
        updateSelectedEfosList();
        updateGeneList();
    };

    init();
});
// --- Dark Mode ---
    const darkModeToggle = document.getElementById('dark-mode-toggle');

    const enableDarkMode = () => {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'enabled');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
        }
    };

    const disableDarkMode = () => {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'disabled');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
        }
    };

    // Check for saved preference on page load
    if (localStorage.getItem('darkMode') === 'enabled') {
        enableDarkMode();
    } else {
        disableDarkMode();
    }

    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            if (localStorage.getItem('darkMode') !== 'enabled') {
                enableDarkMode();
            } else {
                disableDarkMode();
            }
        });
    }
