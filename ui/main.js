document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const analyzeBtn = document.getElementById('analyze-btn');
  const modeSelect = document.getElementById('mode-select');
  
  const canvas = document.getElementById('result-canvas');
  const ctx = canvas.getContext('2d');
  const imagePreview = document.getElementById('image-preview');
  const placeholderText = document.getElementById('placeholder-text');
  const loader = document.getElementById('loader');
  
  const statsBox = document.getElementById('stats-box');
  const statCount = document.getElementById('stat-count');
  const statClassical = document.getElementById('stat-classical');
  const statMode = document.getElementById('stat-mode');

  let currentFile = null;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  });

  browseBtn.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', function() {
    handleFiles(this.files);
  });

  function handleFiles(files) {
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        currentFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
          imagePreview.src = e.target.result;
          imagePreview.onload = () => {
            canvas.width = imagePreview.naturalWidth;
            canvas.height = imagePreview.naturalHeight;
            ctx.drawImage(imagePreview, 0, 0);
            
            placeholderText.style.display = 'none';
            statsBox.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Image';
          };
        };
        reader.readAsDataURL(file);
      } else {
        alert("Please upload an image file.");
      }
    }
  }

  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    const mode = modeSelect.value;

    analyzeBtn.disabled = true;
    loader.style.display = 'flex';
    statsBox.style.display = 'none';

    try {
      ctx.drawImage(imagePreview, 0, 0);

      const base64Image = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(currentFile);
      });

      const payload = {
        mode: mode,
        image: base64Image
      };

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      
      drawResults(data);
      
      statCount.textContent = data.count;
      statClassical.textContent = data.classical_count;
      statMode.textContent = data.mode;
      statsBox.style.display = 'block';
      
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to analyze image. Ensure the backend is running on port 8000.");
    } finally {
      loader.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  function drawResults(data) {
    const { boxes, scores } = data;
    
    ctx.lineWidth = Math.max(2, canvas.width / 400);
    ctx.strokeStyle = '#10b981';
    ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
    
    const fontHeight = Math.max(12, canvas.width / 50);
    ctx.font = `600 ${fontHeight}px Inter, sans-serif`;
    
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const score = scores[i];
      
      const x = box[0];
      const y = box[1];
      const width = box[2] - box[0];
      const height = box[3] - box[1];
      
      ctx.beginPath();
      ctx.rect(x, y, width, height);
      ctx.fill();
      ctx.stroke();
      
      if (score !== undefined) {
        const text = `${(score * 100).toFixed(0)}%`;
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        
        ctx.fillStyle = '#10b981';
        ctx.fillRect(x - ctx.lineWidth/2, y - fontHeight - 4, textWidth + 8, fontHeight + 4);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x + 4, y - 4);
        
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
      }
    }
  }
});
