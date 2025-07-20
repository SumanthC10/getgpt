document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const diseaseQueryInput = document.getElementById('disease-query');
    const topKSelect = document.getElementById('top-k-select');
    const efoResultsList = document.getElementById('efo-results-list');
    const getGenesButton = document.getElementById('get-genes-button');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    const loader = document.getElementById('loader');

    let geneDataTable;

    // --- Utility Functions ---
    const showLoader = () => loader.style.display = 'flex';
    const hideLoader = () => loader.style.display = 'none';

    // --- Event Listeners ---
    searchBtn.addEventListener('click', async () => {
        const query = diseaseQueryInput.value.trim();
        const topK = topKSelect.value;
        if (!query) return;

        showLoader();
        try {
            const response = await fetch(`${window.API_BASE_URL}/v1/efo_search?q=${query}&top_k=${topK}`);
            const data = await response.json();
            
            efoResultsList.innerHTML = '';
            data.results.forEach(item => {
                const li = document.createElement('li');
                li.textContent = `${item.label} (${item.efo_id}) - Score: ${item.score.toFixed(2)}`;
                li.dataset.efoId = item.efo_id;
                li.dataset.score = item.score;
                li.addEventListener('click', () => {
                    li.classList.toggle('selected');
                    const selectedCount = efoResultsList.querySelectorAll('.selected').length;
                    getGenesButton.style.display = selectedCount > 0 ? 'block' : 'none';
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
        const selectedItems = efoResultsList.querySelectorAll('.selected');
        if (selectedItems.length === 0) return;

        showLoader();
        try {
            const allGenes = [];
            for (const item of selectedItems) {
                const efoId = item.dataset.efoId;
                const response = await fetch(`${window.API_BASE_URL}/v1/get-list`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ disease_id: efoId })
                });
                const data = await response.json();
                data.results.forEach(gene => {
                    gene.efo_id = efoId; // Add EFO ID to each gene
                    allGenes.push(gene);
                });
            }
            populateGeneTable(allGenes);
            downloadCsvBtn.style.display = 'block';
        } catch (error) {
            console.error("Error fetching gene list:", error);
        } finally {
            hideLoader();
        }
    });

    downloadCsvBtn.addEventListener('click', () => {
        const tableData = geneDataTable.rows().data().toArray();
        let csvContent = "data:text/csv;charset=utf-8,Gene,Source,Score,Evidence,EFO ID\n";
        
        tableData.forEach(rowArray => {
            const row = rowArray.map(cell => `"${cell.toString().replace(/"/g, '""')}"`).join(",");
            csvContent += row + "\n";
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "gene_list.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // --- Table Functions ---
    function populateGeneTable(genes) {
        if (geneDataTable) {
            geneDataTable.destroy();
        }

        const tableBody = document.querySelector('#gene-table tbody');
        tableBody.innerHTML = ''; // Clear existing data

        genes.forEach(gene => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${gene.gene_symbol}</td>
                <td>${Array.isArray(gene.source) ? gene.source.join(', ') : gene.source}</td>
                <td>${gene.overall_score.toFixed(4)}</td>
                <td>${formatEvidence(gene.evidence)}</td>
                <td>${gene.efo_id}</td>
            `;
            tableBody.appendChild(row);
        });

        geneDataTable = $('#gene-table').DataTable({
            "pageLength": 10,
            "responsive": true,
            "destroy": true
        });
    }

    function formatEvidence(evidence) {
        if (!evidence || evidence.length === 0) return 'N/A';
        if (typeof evidence[0] === 'string') {
            return evidence.join(', ');
        }
        if (typeof evidence[0] === 'object' && evidence[0] !== null) {
            return evidence.map(item => item.rsid || JSON.stringify(item)).join(', ');
        }
        return JSON.stringify(evidence);
    }

    // --- Initialization ---
});
// --- Dark Mode ---
    const darkModeToggle = document.getElementById('dark-mode-toggle');

    const enableDarkMode = () => {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'enabled');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = '&#9790;'; // Moon icon
        }
    };

    const disableDarkMode = () => {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'disabled');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = '&#9728;'; // Sun icon
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
