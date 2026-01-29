document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analysisInterface = document.getElementById('analysis-interface');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const verdictLabel = document.getElementById('verdict-label');
    const confidenceValue = document.getElementById('confidence-value');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    // Verification Elements
    const verifyYesBtn = document.getElementById('verify-yes-btn');
    const verifyNoBtn = document.getElementById('verify-no-btn');
    const verificationSection = document.getElementById('verification-section');
    const saveMsg = document.getElementById('save-msg');

    let currentFile = null;
    let currentPrediction = null; // Store prediction for verification

    // ... (Event Listeners for Drag/Drop remain similar) ...
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            showPreview(currentFile);
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            // Switch View
            dropZone.style.display = 'none';
            analysisInterface.style.display = 'block';

            // Reset Result Side
            resultPlaceholder.style.display = 'block';
            resultContent.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        currentFile = null;
        currentPrediction = null;
        fileInput.value = '';
        analysisInterface.style.display = 'none';
        dropZone.style.display = 'block';

        // Reset Verification UI
        if (verificationSection) {
            verificationSection.style.display = 'none'; // Hide verification section on reset
        }
        if (verifyYesBtn && verifyNoBtn) {
            verifyYesBtn.disabled = false;
            verifyNoBtn.disabled = false;
        }
        if (saveMsg) saveMsg.textContent = '';

        if (myChart) {
            myChart.destroy();
            myChart = null;
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Loading State
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            loader.style.display = 'none';
        }
    });

    let myChart = null;

    function displayResult(data) {
        currentPrediction = data; // Store for verification

        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';

        verdictLabel.textContent = data.label;
        if (data.label === 'AI Generated') {
            verdictLabel.className = 'verdict is-ai';
        } else {
            verdictLabel.className = 'verdict is-real';
        }

        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;

        // Render Chart
        const ctx = document.getElementById('confidenceChart').getContext('2d');

        if (myChart) {
            myChart.destroy();
        }

        const probReal = data.raw_score;
        const probAI = 1 - data.raw_score;

        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Real', 'AI Generated'],
                datasets: [{
                    label: 'Probability',
                    data: [probReal, probAI],
                    backgroundColor: [
                        'rgba(56, 142, 60, 0.6)',
                        'rgba(211, 47, 47, 0.6)'
                    ],
                    borderColor: [
                        'rgba(56, 142, 60, 1)',
                        'rgba(211, 47, 47, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1,
                        ticks: { callback: (val) => (val * 100) + '%' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return (context.raw * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        // Enable verification buttons after analysis
        if (verificationSection) {
            verificationSection.style.display = 'flex';
        }
        if (verifyYesBtn && verifyNoBtn) {
            verifyYesBtn.disabled = false;
            verifyNoBtn.disabled = false;
            saveMsg.textContent = ''; // Clear any previous save messages
        }
    }

    // Simplified Verification Logic (Yes/No)
    function handleVerification(isCorrect) {
        if (!currentFile || !currentPrediction) return;

        let groundTruth;
        if (isCorrect) {
            groundTruth = currentPrediction.label; // User agrees
        } else {
            // User disagrees, flip label
            groundTruth = (currentPrediction.label === 'AI Generated') ? 'Real' : 'AI Generated';
        }

        // Disable and HIDE buttons
        if (verificationSection) {
            verificationSection.style.display = 'none';
        }

        saveMsg.textContent = "Saving...";
        saveMsg.style.color = "#666";

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('ground_truth', groundTruth);

        fetch('/api/save-test-case', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    saveMsg.textContent = "Saved to Test Suite!";
                    saveMsg.style.color = "var(--success-color)";
                } else {
                    saveMsg.textContent = "Error saving.";
                    saveMsg.style.color = "var(--danger-color)";
                    verifyYesBtn.disabled = false;
                    verifyNoBtn.disabled = false;
                }
            })
            .catch(err => {
                console.error(err);
                saveMsg.textContent = "Error saving.";
                saveMsg.style.color = "var(--danger-color)";
                verifyYesBtn.disabled = false;
                verifyNoBtn.disabled = false;
            });
    }

    if (verifyYesBtn && verifyNoBtn) {
        verifyYesBtn.addEventListener('click', () => handleVerification(true));
        verifyNoBtn.addEventListener('click', () => handleVerification(false));
    }
});
