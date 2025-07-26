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
            
            data.results.forEach(item => {
                // Prevent duplicates
                if (document.querySelector(`li[data-efo-id="${item.efo_id}"]`)) {
                    return;
                }
                const li = document.createElement('li');
                li.textContent = `${item.label} (${item.efo_id}) - Score: ${item.score.toFixed(2)}`;
                li.dataset.efoId = item.efo_id;
                li.dataset.score = item.score;
                li.addEventListener('click', () => {
                    const selectedCount = efoResultsList.querySelectorAll('.selected').length;
                    if (!li.classList.contains('selected') && selectedCount >= 5) {
                        alert("You can select a maximum of 5 EFO IDs.");
                        return;
                    }
                    li.classList.toggle('selected');
                    const newSelectedCount = efoResultsList.querySelectorAll('.selected').length;
                    getGenesButton.style.display = newSelectedCount > 0 ? 'block' : 'none';
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
                <td>${formatSource(gene.source)}</td>
                <td>${gene.g_score ? gene.g_score.toFixed(4) : ''}</td>
                <td>${gene.e_score ? gene.e_score.toFixed(4) : ''}</td>
                <td>${gene.t_score ? gene.t_score.toFixed(4) : ''}</td>
                <td>${gene.overall_score.toFixed(4)}</td>
                <td>${formatEvidence(gene.evidence)}</td>
                <td><a href="http://purl.obolibrary.org/obo/${gene.efo_id.replace(':', '_')}" target="_blank" class="efo-link">${gene.efo_id}</a></td>
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
        if (!evidence || evidence.length === 0) return '';

        // Check if evidence is in the new hyperlink format
        if (typeof evidence[0] === 'object' && evidence[0] !== null && 'url' in evidence[0]) {
            return evidence.map(item => `<a href="${item.url}" target="_blank" class="evidence-link">${item.display}</a>`).join(' ');
        }
        
        // Fallback for old format
        if (typeof evidence[0] === 'string') {
            return evidence.join(', ');
        }
        if (typeof evidence[0] === 'object' && evidence[0] !== null) {
            return evidence.map(item => item.rsid || JSON.stringify(item)).join(', ');
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
