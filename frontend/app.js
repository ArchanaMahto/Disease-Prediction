document.addEventListener('DOMContentLoaded', () => {
    // State
    let allSymptoms = [];
    let selectedSymptoms = new Set();
    let benchmarksData = null;

    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const symptomSearch = document.getElementById('symptomSearch');
    const autocompleteDropdown = document.getElementById('autocompleteDropdown');
    const selectedTagsTray = document.getElementById('selectedTagsTray');
    const symptomCountBadge = document.getElementById('symptomCountBadge');
    const clearSymptomsBtn = document.getElementById('clearSymptomsBtn');
    const modelSelect = document.getElementById('modelSelect');
    const predictBtn = document.getElementById('predictBtn');
    const predictSpinner = document.getElementById('predictSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const activeModelTag = document.getElementById('activeModelTag');
    const benchmarkTableBody = document.getElementById('benchmarkTableBody');
    const refreshBenchmarksBtn = document.getElementById('refreshBenchmarksBtn');
    const errorToast = document.getElementById('errorToast');
    const toastMessage = document.getElementById('toastMessage');

    // 1. Initialize API Fetching
    init();

    async function init() {
        setupTabs();
        setupEvents();
        await fetchSymptoms();
        await fetchBenchmarks();
    }

    // Tab Navigation
    function setupTabs() {
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                
                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                document.getElementById(targetTab).classList.add('active');

                if (targetTab === 'benchmarkTab') {
                    renderBenchmarks();
                }
            });
        });
    }

    // Fetch Symptoms Metadata
    async function fetchSymptoms() {
        try {
            const res = await fetch('/api/symptoms');
            if (!res.ok) throw new Error('Failed to fetch symptoms catalog.');
            const data = await res.json();
            allSymptoms = data.symptoms || [];
        } catch (err) {
            showToast(err.message || 'Error loading symptoms.');
        }
    }

    // Fetch Benchmarks
    async function fetchBenchmarks() {
        try {
            const res = await fetch('/api/models/compare');
            if (!res.ok) throw new Error('Failed to fetch benchmark metrics.');
            const data = await res.json();
            benchmarksData = data.models || {};
        } catch (err) {
            console.warn('Could not load benchmarks:', err);
        }
    }

    // Event Handlers
    function setupEvents() {
        // Search Input Filtering
        symptomSearch.addEventListener('input', (e) => {
            const query = e.target.value.trim().toLowerCase();
            if (!query) {
                hideAutocomplete();
                return;
            }

            const matches = allSymptoms.filter(s => 
                !selectedSymptoms.has(s.id) && 
                s.name.toLowerCase().includes(query)
            );

            renderAutocomplete(matches);
        });

        // Click outside autocomplete
        document.addEventListener('click', (e) => {
            if (!symptomSearch.contains(e.target) && !autocompleteDropdown.contains(e.target)) {
                hideAutocomplete();
            }
        });

        // Clear All
        clearSymptomsBtn.addEventListener('click', () => {
            selectedSymptoms.clear();
            updateSelectedUI();
            resetResults();
        });

        // Model Switcher Event
        modelSelect.addEventListener('change', () => {
            const selectedOptionText = modelSelect.options[modelSelect.selectedIndex].text;
            activeModelTag.textContent = selectedOptionText.split(' (')[0];
            
            if (selectedSymptoms.size > 0) {
                runPrediction();
            }
        });

        // Predict Button Click
        predictBtn.addEventListener('click', runPrediction);

        // Refresh Benchmarks
        refreshBenchmarksBtn.addEventListener('click', async () => {
            await fetchBenchmarks();
            renderBenchmarks();
        });
    }

    // Autocomplete Dropdown Rendering
    function renderAutocomplete(items) {
        if (items.length === 0) {
            autocompleteDropdown.innerHTML = `<div class="autocomplete-item text-muted">No matching symptoms found</div>`;
        } else {
            autocompleteDropdown.innerHTML = items.map(item => `
                <div class="autocomplete-item" data-id="${item.id}" data-name="${item.name}">
                    ${item.name}
                </div>
            `).join('');

            autocompleteDropdown.querySelectorAll('.autocomplete-item').forEach(el => {
                el.addEventListener('click', () => {
                    const id = el.getAttribute('data-id');
                    if (id) {
                        selectedSymptoms.add(id);
                        symptomSearch.value = '';
                        hideAutocomplete();
                        updateSelectedUI();
                    }
                });
            });
        }
        autocompleteDropdown.classList.remove('hidden');
    }

    function hideAutocomplete() {
        autocompleteDropdown.classList.add('hidden');
    }

    // Update Selected Symptoms Tags UI
    function updateSelectedUI() {
        symptomCountBadge.textContent = `${selectedSymptoms.size} selected`;
        predictBtn.disabled = selectedSymptoms.size === 0;

        if (selectedSymptoms.size === 0) {
            selectedTagsTray.innerHTML = `<p class="placeholder-text">No symptoms selected yet. Search above to add.</p>`;
            return;
        }

        selectedTagsTray.innerHTML = Array.from(selectedSymptoms).map(id => {
            const sym = allSymptoms.find(s => s.id === id);
            const name = sym ? sym.name : id;
            return `
                <div class="tag-chip">
                    <span>${name}</span>
                    <span class="tag-remove" data-id="${id}">&times;</span>
                </div>
            `;
        }).join('');

        selectedTagsTray.querySelectorAll('.tag-remove').forEach(el => {
            el.addEventListener('click', () => {
                const id = el.getAttribute('data-id');
                selectedSymptoms.delete(id);
                updateSelectedUI();
                if (selectedSymptoms.size === 0) {
                    resetResults();
                }
            });
        });
    }

    // Reset Results View
    function resetResults() {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📋</div>
                <h3>No Symptoms Analyzed</h3>
                <p>Select symptoms from the left panel and click "Run Differential Diagnosis" to generate probability estimates.</p>
            </div>
        `;
    }

    // Execute Prediction API Call
    async function runPrediction() {
        if (selectedSymptoms.size === 0) return;

        setLoading(true);
        const payload = {
            symptoms: Array.from(selectedSymptoms),
            model_name: modelSelect.value,
            top_n: 5
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to generate prediction.');
            }

            const data = await res.json();
            renderPredictionResults(data);
        } catch (err) {
            showToast(err.message || 'Prediction failed.');
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        predictBtn.disabled = isLoading;
        if (isLoading) {
            predictSpinner.classList.remove('hidden');
        } else {
            predictSpinner.classList.add('hidden');
        }
    }

    // Render Differential Diagnosis Cards
    function renderPredictionResults(data) {
        if (!data.predictions || data.predictions.length === 0) {
            resultsContainer.innerHTML = `<p class="placeholder-text">No predictions returned.</p>`;
            return;
        }

        resultsContainer.innerHTML = data.predictions.map(pred => `
            <div class="prediction-card">
                <div class="pred-header">
                    <div class="pred-title">
                        <span class="rank-badge">#${pred.rank}</span>
                        <span>${pred.disease}</span>
                    </div>
                    <span class="confidence-val">${pred.confidence_percentage}%</span>
                </div>

                <div class="meter-bg">
                    <div class="meter-fill" style="width: ${pred.confidence_percentage}%;"></div>
                </div>

                <div class="xai-section">
                    <div class="xai-title">Key Contributing Symptoms:</div>
                    <div class="attribution-list">
                        ${pred.contributing_symptoms.length > 0 ? pred.contributing_symptoms.map(attr => `
                            <span class="attr-pill">
                                ${attr.name}
                                <span class="attr-weight">${Math.round(attr.importance_score * 100)}%</span>
                            </span>
                        `).join('') : '<span class="attr-pill">General Profile</span>'}
                    </div>
                </div>
            </div>
        `).join('');
    }

    // Render Benchmarks Table
    function renderBenchmarks() {
        if (!benchmarksData) {
            benchmarkTableBody.innerHTML = `<tr><td colspan="7" class="table-loading">No benchmark data available. Run main.py training script.</td></tr>`;
            return;
        }

        benchmarkTableBody.innerHTML = Object.entries(benchmarksData).map(([key, item]) => `
            <tr>
                <td><strong>${item.name}</strong></td>
                <td><span class="badge">${(item.accuracy * 100).toFixed(1)}%</span></td>
                <td>${(item.precision * 100).toFixed(1)}%</td>
                <td>${(item.recall * 100).toFixed(1)}%</td>
                <td><strong>${(item.f1_score * 100).toFixed(1)}%</strong></td>
                <td>${item.latency_ms} ms</td>
                <td>${item.train_time_sec} s</td>
            </tr>
        `).join('');
    }

    // Toast Utility
    function showToast(message) {
        toastMessage.textContent = message;
        errorToast.classList.remove('hidden');
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 4000);
    }
});
