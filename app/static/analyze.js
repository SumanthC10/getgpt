document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const genesetSelectionContainer = document.getElementById('geneset-selection-container');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analysisResultsContainer = document.getElementById('analysis-results-container');
    const loader = document.getElementById('loader');

    // --- State Management ---
    let geneSets = JSON.parse(sessionStorage.getItem('geneSets')) || {};
    let selectedEfos = JSON.parse(sessionStorage.getItem('selectedEfos')) || [];

    // --- Utility Functions ---
    const showLoader = () => loader.style.display = 'flex';
    const hideLoader = () => loader.style.display = 'none';

    // --- UI Update Functions ---
    const populateGeneSetSelection = () => {
        genesetSelectionContainer.innerHTML = '';
        if (Object.keys(geneSets).length === 0) {
            genesetSelectionContainer.innerHTML = '<p>No gene sets found. Please select EFOs and get genes on the homepage first.</p>';
            analyzeBtn.style.display = 'none';
            return;
        }

        for (const efoId in geneSets) {
            const efo = selectedEfos.find(e => e.efo_id === efoId);
            const label = efo ? efo.label : efoId;
            const checkbox = document.createElement('div');
            checkbox.innerHTML = `
                <input type="checkbox" id="${efoId}" name="geneset" value="${efoId}">
                <label for="${efoId}">${label} (${efoId})</label>
            `;
            genesetSelectionContainer.appendChild(checkbox);
        }
    };

    // --- Event Listeners ---
    analyzeBtn.addEventListener('click', async () => {
        const selectedCheckboxes = document.querySelectorAll('input[name="geneset"]:checked');
        if (selectedCheckboxes.length === 0) {
            alert('Please select at least one gene set to analyze.');
            return;
        }

        let allGenes = new Set();
        selectedCheckboxes.forEach(checkbox => {
            const efoId = checkbox.value;
            if (geneSets[efoId]) {
                geneSets[efoId].forEach(gene => allGenes.add(gene.gene_symbol));
            }
        });

        const geneList = Array.from(allGenes);
        if (geneList.length === 0) {
            alert('The selected gene sets are empty.');
            return;
        }

        showLoader();
        try {
            const response = await fetch(`${window.API_BASE_URL}/v1/genes/pager`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gene_list: geneList })
            });
            const data = await response.json();
            displayAnalysisResults(data.results);
        } catch (error) {
            console.error('Error performing PAGER analysis:', error);
            analysisResultsContainer.innerHTML = '<p>An error occurred during analysis. Please try again.</p>';
        } finally {
            hideLoader();
        }
    });

    const displayAnalysisResults = (results) => {
        analysisResultsContainer.innerHTML = '';
        if (!results || results.length === 0) {
            analysisResultsContainer.innerHTML = '<p>No significant pathways found.</p>';
            return;
        }

        const numPags = results.length;
        analysisResultsContainer.innerHTML += `<h3>Found ${numPags} significant pathways (PAGs).</h3>`;

        const resultsToShow = results.slice(0, 15);
        resultsToShow.forEach(pag => {
            const pagElement = document.createElement('div');
            pagElement.className = 'pag-result';

            const genesToShow = pag.genes.slice(0, 10).join(', ');
            pagElement.innerHTML = `
                <h4>${pag.pag_name}</h4>
                <p><strong>Overlap:</strong> ${pag.overlap}</p>
                <p><strong>P-value:</strong> ${pag.p_value.toExponential(2)}</p>
                <p><strong>Genes:</strong> ${genesToShow}</p>
            `;
            analysisResultsContainer.appendChild(pagElement);
        });
    };

    // --- Initialization ---
    const init = () => {
        populateGeneSetSelection();
    };

    init();
});