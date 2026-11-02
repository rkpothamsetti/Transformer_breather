/**
 * Silica Gel Phase & Color Detector - Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');
    const btnClear = document.getElementById('btn-clear');
    
    const loadingBox = document.getElementById('loading-box');
    const resultCard = document.getElementById('result-card');
    
    const valGelType = document.getElementById('val-gel-type');
    const valPhase = document.getElementById('val-phase');
    const valColorCircle = document.getElementById('val-color-circle');
    const valColorName = document.getElementById('val-color-name');

    // Click dropzone to open file dialog
    dropzone.addEventListener('click', (e) => {
        if (e.target !== btnClear && !previewBox.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-active');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clipboard Paste (Ctrl+V)
    window.addEventListener('paste', (e) => {
        if (e.clipboardData && e.clipboardData.items) {
            for (let item of e.clipboardData.items) {
                if (item.type.startsWith('image/')) {
                    handleFile(item.getAsFile());
                    break;
                }
            }
        }
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select an image file (PNG, JPG, JPEG).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            // Show preview
            previewImg.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            previewBox.classList.remove('hidden');

            // Automatically analyze
            analyzePhoto(file);
        };
        reader.readAsDataURL(file);
    }

    async function analyzePhoto(file) {
        loadingBox.classList.remove('hidden');
        resultCard.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                throw new Error(data.error || 'Failed to analyze photo');
            }

            renderOutput(data.result);
        } catch (err) {
            console.error('Error:', err);
            alert(`Analysis failed: ${err.message}`);
        } finally {
            loadingBox.classList.add('hidden');
        }
    }

    function renderOutput(result) {
        resultCard.classList.remove('hidden');

        // 1. Gel Type
        valGelType.textContent = `${result.gel_type} Silica Gel`;

        // 2. Current Phase (Unsaturated / Semi Saturated / Saturated)
        let phaseClean = 'Unsaturated';
        let phaseClass = 'phase-unsaturated';
        const condLower = result.condition.toLowerCase();

        if (condLower.includes('semi')) {
            phaseClean = 'Semi Saturated';
            phaseClass = 'phase-semi-saturated';
        } else if (condLower.includes('saturated') && !condLower.includes('unsaturated')) {
            phaseClean = 'Saturated';
            phaseClass = 'phase-saturated';
        } else {
            phaseClean = 'Unsaturated';
            phaseClass = 'phase-unsaturated';
        }

        valPhase.textContent = phaseClean;
        valPhase.className = `value phase-badge ${phaseClass}`;

        // 3. Current Colour
        const hex = result.dominant_color.hex;
        valColorCircle.style.backgroundColor = hex;

        // Friendly color descriptor based on Gel and Phase
        let colorDesc = result.characteristic_color || '';
        valColorName.textContent = `${colorDesc} (${hex})`;
    }

    function resetUI() {
        fileInput.value = '';
        previewImg.src = '';
        uploadPrompt.classList.remove('hidden');
        previewBox.classList.add('hidden');
        loadingBox.classList.add('hidden');
        resultCard.classList.add('hidden');
    }
});
