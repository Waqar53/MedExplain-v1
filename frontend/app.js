/**
 * MedExplain - Medical Report Analysis System
 * 
 * Frontend application for healthcare report processing
 * Version: 1.0.0
 */

(function () {
    'use strict';

    // Configuration
    const CONFIG = {
        API_BASE_URL: 'http://localhost:8000',
        MAX_FILE_SIZE_MB: 10,
        VALID_REPORT_EXTENSIONS: ['.pdf', '.txt'],
        VALID_IMAGE_EXTENSIONS: ['.png', '.jpg', '.jpeg']
    };

    // Application State
    const state = {
        currentFile: null,
        currentFileType: 'report',
        sessionId: null,
        isProcessing: false
    };

    // DOM Element References
    const elements = {};

    /**
     * Initialize application on DOM ready
     */
    function init() {
        cacheElements();
        bindEvents();
        console.log('MedExplain System initialized');
    }

    /**
     * Cache DOM element references
     */
    function cacheElements() {
        elements.tabBtns = document.querySelectorAll('.tab-btn');
        elements.fileInput = document.getElementById('fileInput');
        elements.dropzone = document.getElementById('dropzone');
        elements.filePreview = document.getElementById('filePreview');
        elements.previewIcon = document.getElementById('previewIcon');
        elements.fileName = document.getElementById('fileName');
        elements.fileSize = document.getElementById('fileSize');
        elements.removeFile = document.getElementById('removeFile');
        elements.analyzeBtn = document.getElementById('analyzeBtn');
        elements.contextSection = document.getElementById('contextSection');
        elements.additionalContext = document.getElementById('additionalContext');
        elements.uploadSection = document.getElementById('uploadSection');
        elements.processingSection = document.getElementById('processingSection');
        elements.resultsSection = document.getElementById('resultsSection');
        elements.errorSection = document.getElementById('errorSection');
        elements.processingStep = document.getElementById('processingStep');
        elements.progressFill = document.getElementById('progressFill');
        elements.confidenceLevel = document.getElementById('confidenceLevel');
        elements.confidenceFill = document.getElementById('confidenceFill');
        elements.riskLevel = document.getElementById('riskLevel');
        elements.summaryText = document.getElementById('summaryText');
        elements.meaningText = document.getElementById('meaningText');
        elements.findingsList = document.getElementById('findingsList');
        elements.stepsList = document.getElementById('stepsList');
        elements.mainDisclaimer = document.getElementById('mainDisclaimer');
        elements.consultationReminder = document.getElementById('consultationReminder');
        elements.confidenceNote = document.getElementById('confidenceNote');
        elements.downloadPdf = document.getElementById('downloadPdf');
        elements.newAnalysis = document.getElementById('newAnalysis');
        elements.retryBtn = document.getElementById('retryBtn');
        elements.errorMessage = document.getElementById('errorMessage');
        elements.reportTypes = document.querySelector('.report-types');
        elements.xrayTypes = document.querySelector('.xray-types');
    }

    /**
     * Bind event listeners
     */
    function bindEvents() {
        // Tab navigation
        elements.tabBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                handleTabSwitch(this.dataset.tab);
            });
        });

        // File input
        elements.fileInput.addEventListener('change', handleFileSelect);

        // Drag and drop
        elements.dropzone.addEventListener('dragover', handleDragOver);
        elements.dropzone.addEventListener('dragleave', handleDragLeave);
        elements.dropzone.addEventListener('drop', handleDrop);
        elements.dropzone.addEventListener('click', function () {
            elements.fileInput.click();
        });

        // File management
        elements.removeFile.addEventListener('click', function (e) {
            e.stopPropagation();
            resetFileSelection();
        });

        // Analysis
        elements.analyzeBtn.addEventListener('click', processReport);

        // Results actions
        elements.downloadPdf.addEventListener('click', exportPdf);
        elements.newAnalysis.addEventListener('click', resetApplication);
        elements.retryBtn.addEventListener('click', resetApplication);
    }

    /**
     * Handle tab switching between report types
     */
    function handleTabSwitch(tab) {
        state.currentFileType = tab;

        elements.tabBtns.forEach(function (btn) {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        if (tab === 'report') {
            elements.fileInput.accept = CONFIG.VALID_REPORT_EXTENSIONS.join(',');
            elements.reportTypes.style.display = 'inline';
            elements.xrayTypes.style.display = 'none';
        } else {
            elements.fileInput.accept = CONFIG.VALID_IMAGE_EXTENSIONS.join(',');
            elements.reportTypes.style.display = 'none';
            elements.xrayTypes.style.display = 'inline';
        }

        if (state.currentFile) {
            resetFileSelection();
        }
    }

    /**
     * Handle file selection from input
     */
    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            validateAndSetFile(e.target.files[0]);
        }
    }

    /**
     * Handle drag over event
     */
    function handleDragOver(e) {
        e.preventDefault();
        elements.dropzone.classList.add('dragover');
    }

    /**
     * Handle drag leave event
     */
    function handleDragLeave(e) {
        e.preventDefault();
        elements.dropzone.classList.remove('dragover');
    }

    /**
     * Handle file drop event
     */
    function handleDrop(e) {
        e.preventDefault();
        elements.dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            validateAndSetFile(e.dataTransfer.files[0]);
        }
    }

    /**
     * Validate file and set as current
     */
    function validateAndSetFile(file) {
        var extension = '.' + file.name.split('.').pop().toLowerCase();
        var validExtensions = state.currentFileType === 'report'
            ? CONFIG.VALID_REPORT_EXTENSIONS
            : CONFIG.VALID_IMAGE_EXTENSIONS;

        if (validExtensions.indexOf(extension) === -1) {
            displayError('Invalid file format. Accepted formats: ' + validExtensions.join(', '));
            return;
        }

        var maxSizeBytes = CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            displayError('File exceeds maximum size of ' + CONFIG.MAX_FILE_SIZE_MB + 'MB.');
            return;
        }

        state.currentFile = file;
        displayFilePreview(file);
    }

    /**
     * Display file preview after selection
     */
    function displayFilePreview(file) {
        var extension = file.name.split('.').pop().toLowerCase();
        var iconMap = {
            'pdf': 'PDF',
            'txt': 'TXT',
            'png': 'IMG',
            'jpg': 'IMG',
            'jpeg': 'IMG'
        };

        elements.previewIcon.textContent = iconMap[extension] || 'FILE';
        elements.fileName.textContent = file.name;
        elements.fileSize.textContent = formatFileSize(file.size);

        elements.dropzone.style.display = 'none';
        elements.filePreview.style.display = 'flex';
        elements.contextSection.style.display = 'block';
    }

    /**
     * Reset file selection
     */
    function resetFileSelection() {
        state.currentFile = null;
        elements.fileInput.value = '';
        elements.dropzone.style.display = 'block';
        elements.filePreview.style.display = 'none';
        elements.contextSection.style.display = 'none';
        elements.additionalContext.value = '';
    }

    /**
     * Format file size for display
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        var k = 1024;
        var sizes = ['Bytes', 'KB', 'MB', 'GB'];
        var i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Process the uploaded report
     */
    async function processReport() {
        if (!state.currentFile || state.isProcessing) {
            return;
        }

        state.isProcessing = true;
        showSection('processing');
        updateProgress('Uploading file...', 10);

        try {
            // Upload file
            var uploadResult = await uploadFile(state.currentFile);
            state.sessionId = uploadResult.session_id;

            updateProgress('Analyzing content...', 40);

            // Generate analysis
            var analysisResult = await generateAnalysis(state.sessionId);

            updateProgress('Preparing results...', 90);
            await delay(300);

            // Display results
            renderResults(analysisResult);
            showSection('results');

        } catch (error) {
            console.error('Processing error:', error);
            elements.errorMessage.textContent = error.message || 'An error occurred during processing. Please try again.';
            showSection('error');
        } finally {
            state.isProcessing = false;
        }
    }

    /**
     * Upload file to server
     */
    async function uploadFile(file) {
        var formData = new FormData();
        formData.append('file', file);

        var endpoint = state.currentFileType === 'report' ? '/upload-report' : '/upload-xray';

        var response = await fetch(CONFIG.API_BASE_URL + endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            var error = await response.json();
            throw new Error(error.detail || 'File upload failed');
        }

        return response.json();
    }

    /**
     * Generate analysis from uploaded file
     */
    async function generateAnalysis(sessionId) {
        var context = elements.additionalContext.value;

        var response = await fetch(CONFIG.API_BASE_URL + '/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                additional_context: context || null
            })
        });

        if (!response.ok) {
            var error = await response.json();
            throw new Error(error.detail || 'Analysis generation failed');
        }

        return response.json();
    }

    /**
     * Render analysis results
     */
    function renderResults(data) {
        // Confidence level
        var confidence = data.confidence || 'medium';
        var score = data.confidence_score || 0.5;
        elements.confidenceLevel.innerHTML = '<span class="confidence-badge ' + confidence + '">' + confidence.toUpperCase() + '</span>';
        elements.confidenceFill.style.width = (score * 100) + '%';

        // Priority level
        var risk = data.risk_level || 'low';
        var riskLabels = { 'low': 'ROUTINE', 'medium': 'MODERATE', 'high': 'ELEVATED' };
        elements.riskLevel.innerHTML = '<span class="risk-badge ' + risk + '">' + (riskLabels[risk] || risk.toUpperCase()) + '</span>';

        // Content
        var explanation = data.explanation || {};

        elements.summaryText.textContent = explanation.summary || 'Report analysis complete.';
        elements.meaningText.textContent = explanation.what_this_means || 'Please consult your healthcare provider for interpretation.';

        // Findings
        var findings = explanation.key_findings || [];
        if (findings.length > 0) {
            elements.findingsList.innerHTML = findings.map(function (f) {
                return '<li>' + escapeHtml(f) + '</li>';
            }).join('');
        } else {
            elements.findingsList.innerHTML = '<li>No specific findings identified</li>';
        }

        // Recommendations
        var steps = explanation.common_next_steps || [
            'Review results with your healthcare provider',
            'Discuss any concerns during your next appointment',
            'Follow provider recommendations'
        ];
        elements.stepsList.innerHTML = steps.map(function (s) {
            return '<li>' + escapeHtml(s) + '</li>';
        }).join('');

        // Disclaimers
        var disclaimer = data.disclaimer || {};
        elements.mainDisclaimer.textContent = disclaimer.main_disclaimer || 'This analysis is not a medical diagnosis.';
        elements.consultationReminder.textContent = disclaimer.consultation_reminder || 'Consult a qualified healthcare professional before making any medical decisions.';

        if (disclaimer.confidence_note) {
            elements.confidenceNote.textContent = disclaimer.confidence_note;
            elements.confidenceNote.style.display = 'block';
        } else {
            elements.confidenceNote.style.display = 'none';
        }
    }

    /**
     * Update progress indicator
     */
    function updateProgress(step, percent) {
        elements.processingStep.textContent = step;
        elements.progressFill.style.width = percent + '%';
    }

    /**
     * Show specified section
     */
    function showSection(section) {
        elements.uploadSection.style.display = section === 'upload' ? 'block' : 'none';
        elements.processingSection.style.display = section === 'processing' ? 'block' : 'none';
        elements.resultsSection.style.display = section === 'results' ? 'block' : 'none';
        elements.errorSection.style.display = section === 'error' ? 'block' : 'none';

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /**
     * Reset application to initial state
     */
    function resetApplication() {
        state.currentFile = null;
        state.sessionId = null;
        state.isProcessing = false;
        resetFileSelection();
        elements.progressFill.style.width = '0%';
        showSection('upload');
    }

    /**
     * Export results as PDF
     */
    async function exportPdf() {
        if (!state.sessionId) {
            displayError('No analysis available for export.');
            return;
        }

        try {
            var response = await fetch(CONFIG.API_BASE_URL + '/create-pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.sessionId })
            });

            if (response.ok) {
                var data = await response.json();
                window.open(CONFIG.API_BASE_URL + '/download-pdf/' + data.report_id, '_blank');
            } else {
                displayError('PDF export is currently unavailable. You may use browser print functionality.');
            }
        } catch (error) {
            displayError('PDF export failed. Please try again or use browser print functionality.');
        }
    }

    /**
     * Display error message
     */
    function displayError(message) {
        alert(message);
    }

    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Delay utility
     */
    function delay(ms) {
        return new Promise(function (resolve) {
            setTimeout(resolve, ms);
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
