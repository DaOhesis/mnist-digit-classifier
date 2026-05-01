// Canvas setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas with white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing settings
ctx.strokeStyle = 'black';
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 20;

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoordinates(e);
}

function draw(e) {
    if (!isDrawing) return;

    const [x, y] = getCoordinates(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return [
            (e.touches[0].clientX - rect.left) * scaleX,
            (e.touches[0].clientY - rect.top) * scaleY
        ];
    }

    return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY
    ];
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

// Clear canvas
document.getElementById('clearBtn').addEventListener('click', clearCanvas);

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResults();
}

// Predict button
document.getElementById('predictBtn').addEventListener('click', predictDigit);

async function predictDigit() {
    // Check if canvas is empty
    if (isCanvasEmpty()) {
        showError('Please draw a digit first!');
        return;
    }

    showLoading();
    hideError();

    try {
        // Get canvas image data
        const imageData = canvas.toDataURL('image/png');

        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Error connecting to server: ' + error.message);
    } finally {
        hideLoading();
    }
}

function isCanvasEmpty() {
    const pixelBuffer = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    // Check if all pixels are white (255, 255, 255)
    for (let i = 0; i < pixelBuffer.length; i += 4) {
        if (pixelBuffer[i] !== 255 || pixelBuffer[i + 1] !== 255 || pixelBuffer[i + 2] !== 255) {
            return false;
        }
    }
    return true;
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');

    // Display main prediction
    document.getElementById('prediction').textContent = data.prediction;
    document.getElementById('confidence').textContent =
        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // Display top 3 predictions
    const topPredictionsContainer = document.getElementById('topPredictions');
    topPredictionsContainer.innerHTML = '';

    data.top_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="digit">${pred.digit}</div>
            <div class="bar-container">
                <div class="bar" style="width: ${pred.probability * 100}%"></div>
            </div>
            <div class="probability">${(pred.probability * 100).toFixed(1)}%</div>
        `;
        topPredictionsContainer.appendChild(item);
    });
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

// Prevent scrolling on touch devices when drawing
document.body.addEventListener('touchstart', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

document.body.addEventListener('touchend', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

document.body.addEventListener('touchmove', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });
