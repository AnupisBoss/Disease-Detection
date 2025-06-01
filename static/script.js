// File handling
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const hiddenFileInput = document.getElementById('hiddenFileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const selectedFile = document.getElementById('selectedFile');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFileBtn');
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const processingIndicator = document.getElementById('processingIndicator');
const uploadText = document.getElementById('uploadText');

// File selection handlers
selectFileBtn.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', (e) => {
    if (e.target === selectFileBtn || e.target.closest('.btn-secondary')) return;
    fileInput.click();
});

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        // Hide upload area, show selected file
        uploadArea.style.display = 'none';
        selectedFile.style.display = 'flex';
        
        // Enable analyze button
        analyzeBtn.disabled = false;
        
        // Set the hidden input for form submission
        const dt = new DataTransfer();
        dt.items.add(file);
        hiddenFileInput.files = dt.files;
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Clear file selection
function clearFileSelection() {
    uploadArea.style.display = 'block';
    selectedFile.style.display = 'none';
    analyzeBtn.disabled = true;
    fileInput.value = '';
    hiddenFileInput.value = '';
    uploadText.textContent = 'Drag & drop your chest X-ray image here';
}

removeFileBtn.addEventListener('click', clearFileSelection);

uploadForm.addEventListener('submit', (e) => {
    if (!hiddenFileInput.files.length) {
        e.preventDefault();
        alert('Please select an image file first.');
        return;
    }
    
    analyzeBtn.disabled = true;
    processingIndicator.style.display = 'block';
    selectedFile.style.display = 'none';
});

// History management functions
function deleteHistoryItem(index) {
    if (confirm('Are you sure you want to delete this analysis?')) {
        fetch(`/delete_history/${index}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload the page to reflect the changes
                window.location.reload();
            } else {
                alert('Error deleting item: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error deleting item');
        });
    }
}

function clearAllHistory() {
    if (confirm('Are you sure you want to clear all analysis history?')) {
        fetch('/clear_history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload the page to reflect the changes
                window.location.reload();
            } else {
                alert('Error clearing history: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error clearing history');
        });
    }
}

// Function to get detailed information about each condition
function getConditionInfo(label) {
    const conditionInfo = {
        'COVID': {
            description: 'COVID-19 pneumonia typically shows bilateral ground-glass opacities and consolidations.',
            heatmapExplanation: 'Red/hot areas indicate regions the AI model identified as most characteristic of COVID-19 patterns, often showing peripheral and bilateral lung involvement.',
            clinicalNotes: [
                'Ground-glass opacities are common',
                'Bilateral and peripheral distribution',
                'May progress to consolidation',
                'Often affects lower lobes initially'
            ],
            recommendation: 'This AI analysis suggests COVID-19 patterns. Please consult with a healthcare professional for proper diagnosis and treatment.'
        },
        'Lung_Opacity': {
            description: 'Lung opacity refers to areas of increased density in the lungs that may indicate various conditions.',
            heatmapExplanation: 'Highlighted areas show where the AI detected abnormal opacity patterns that could indicate inflammation, fluid, or other pathological changes.',
            clinicalNotes: [
                'May indicate pneumonia, edema, or inflammation',
                'Can be caused by various factors',
                'Location and pattern are important for diagnosis',
                'May require additional imaging or tests'
            ],
            recommendation: 'Lung opacity detected. Further evaluation by a radiologist or pulmonologist is recommended to determine the underlying cause.'
        },
        'Normal': {
            description: 'The chest X-ray appears normal with clear lung fields and no obvious abnormalities.',
            heatmapExplanation: 'The heatmap shows areas the AI focused on during analysis. Even for normal images, the AI examines typical anatomical structures to confirm normalcy.',
            clinicalNotes: [
                'Clear lung fields observed',
                'Normal cardiac silhouette',
                'No obvious infiltrates or masses',
                'Typical anatomical structures present'
            ],
            recommendation: 'AI analysis suggests normal chest X-ray. However, always consult with healthcare professionals for comprehensive evaluation.'
        },
        'Viral Pneumonia': {
            description: 'Viral pneumonia shows characteristic patterns of lung inflammation caused by viral infection.',
            heatmapExplanation: 'Red/hot areas indicate regions where the AI identified patterns typical of viral pneumonia, often showing diffuse bilateral infiltrates.',
            clinicalNotes: [
                'Bilateral interstitial infiltrates common',
                'May show ground-glass patterns',
                'Often more diffuse than bacterial pneumonia',
                'Can affect both lungs symmetrically'
            ],
            recommendation: 'AI analysis suggests viral pneumonia patterns. Medical evaluation is essential for proper diagnosis and antiviral treatment if indicated.'
        }
    };
    
    return conditionInfo[label] || {
        description: 'Analysis complete.',
        heatmapExplanation: 'The heatmap shows areas of interest identified by the AI model.',
        clinicalNotes: ['Consult healthcare professional for interpretation'],
        recommendation: 'Please consult with a healthcare professional for proper diagnosis.'
    };
}

// Function to get confidence level description
function getConfidenceDescription(confidence) {
    const confidencePercent = confidence * 100;
    
    if (confidencePercent >= 90) {
        return {
            level: 'Very High',
            description: 'The AI model is very confident in this prediction.',
            color: '#27ae60'
        };
    } else if (confidencePercent >= 75) {
        return {
            level: 'High',
            description: 'The AI model shows high confidence in this prediction.',
            color: '#2ecc71'
        };
    } else if (confidencePercent >= 60) {
        return {
            level: 'Moderate',
            description: 'The AI model has moderate confidence. Additional evaluation may be helpful.',
            color: '#f39c12'
        };
    } else if (confidencePercent >= 40) {
        return {
            level: 'Low',
            description: 'The AI model has low confidence. Multiple interpretations are possible.',
            color: '#e67e22'
        };
    } else {
        return {
            level: 'Very Low',
            description: 'The AI model has very low confidence. Results should be interpreted cautiously.',
            color: '#e74c3c'
        };
    }
}

// Enhanced modal functionality
function openModal(originalSrc, heatmapSrc, label, confidence) {
    const conditionInfo = getConditionInfo(label);
    const confidenceInfo = getConfidenceDescription(confidence);
    
    document.getElementById('modalOriginal').src = originalSrc;
    document.getElementById('modalHeatmap').src = heatmapSrc;
    
    document.getElementById('modalInfo').innerHTML = `
        <div class="modal-analysis-info">
            <div class="prediction-section">
                <h2 class="prediction-label ${label.toLowerCase().replace(' ', '_')}">${label}</h2>
                <div class="confidence-section">
                    <div class="confidence-row">
                        <span class="confidence-label">Confidence Level:</span>
                        <span class="confidence-value" style="color: ${confidenceInfo.color}">
                            ${(confidence * 100).toFixed(1)}% (${confidenceInfo.level})
                        </span>
                    </div>
                    <div class="confidence-bar-modal">
                        <div class="confidence-fill-modal ${label.toLowerCase().replace(' ', '_')}" 
                             style="width: ${confidence * 100}%"></div>
                    </div>
                    <p class="confidence-description">${confidenceInfo.description}</p>
                </div>
            </div>
            
            <div class="condition-section">
                <h3><i class="fas fa-info-circle"></i> About This Condition</h3>
                <p class="condition-description">${conditionInfo.description}</p>
            </div>
            
            <div class="heatmap-section">
                <h3><i class="fas fa-fire"></i> Understanding the AI Heatmap</h3>
                <p class="heatmap-explanation">${conditionInfo.heatmapExplanation}</p>
                <div class="heatmap-legend">
                    <div class="legend-item">
                        <div class="legend-color hot"></div>
                        <span>High Importance (Red/Yellow)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color cool"></div>
                        <span>Low Importance (Blue/Purple)</span>
                    </div>
                </div>
            </div>
            
            <div class="clinical-section">
                <h3><i class="fas fa-stethoscope"></i> Clinical Notes</h3>
                <ul class="clinical-notes">
                    ${conditionInfo.clinicalNotes.map(note => `<li>${note}</li>`).join('')}
                </ul>
            </div>
            
            <div class="recommendation-section">
                <h3><i class="fas fa-exclamation-triangle"></i> Important Note</h3>
                <div class="recommendation-box">
                    <p>${conditionInfo.recommendation}</p>
                    <p class="disclaimer"><strong>Disclaimer:</strong> This AI analysis is for educational purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.</p>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('imageModal').style.display = 'block';
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
    document.body.style.overflow = 'auto'; // Restore scrolling
}

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    const modal = document.getElementById('imageModal');
    if (e.target === modal) {
        closeModal();
    }
});

// Set confidence bar widths
document.querySelectorAll('.confidence-fill').forEach(bar => {
    const confidence = parseFloat(bar.dataset.confidence);
    bar.style.width = (confidence * 100) + '%';
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});