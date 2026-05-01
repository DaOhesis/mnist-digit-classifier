# MNIST Digit Classifier Web App

A web application that uses a trained neural network to recognize handwritten digits (0-9) drawn by users.

## Features

- 🎨 Interactive drawing canvas
- 🤖 Real-time digit prediction using trained ANN
- 📊 Confidence scores and top 3 predictions
- 📱 Mobile-friendly touch support
- 🎯 ~97-98% accuracy on MNIST dataset

## Prerequisites

- Python 3.8 or higher
- Trained model file (`mnist_digit_classifier.keras`)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (if not already done):**
   ```bash
   python mnist_digit_solver.py
   ```
   This will create `mnist_digit_classifier.keras` and generate visualization files.

3. **Run the web application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   Navigate to `http://localhost:5000`

## Usage

1. Draw a single digit (0-9) in the canvas
2. Click "Predict" to analyze your drawing
3. View the AI's prediction and confidence score
4. Click "Clear" to try again

## Tips for Best Results

- Draw large, clear digits
- Use a single continuous stroke
- Center the digit in the canvas
- Avoid overlapping lines
- Draw digits similar to standard handwriting

## Project Structure

```
.
├── app.py                      # Flask web application
├── mnist_digit_solver.py       # Model training script
├── predict_digits.py           # Interactive prediction script
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # HTML template
└── static/
    ├── style.css              # CSS styling
    └── script.js              # JavaScript for drawing and API calls
```

## Model Architecture

- Input: 784 features (28×28 flattened images)
- Hidden layers: 512 → 256 → 128 neurons with ReLU activation
- Output: 10 neurons (digits 0-9) with softmax
- Regularization: Dropout (20%)
- Optimizer: Adam
- Training accuracy: ~97-98%

## API Endpoints

### `GET /`
Returns the main web page.

### `POST /predict`
Accepts a base64-encoded image and returns predictions.

**Request:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 7,
  "confidence": 0.9876,
  "top_predictions": [
    {"digit": 7, "probability": 0.9876},
    {"digit": 1, "probability": 0.0089},
    {"digit": 9, "probability": 0.0023}
  ]
}
```

## Troubleshooting

**Model not found:**
- Make sure you've trained the model first using `mnist_digit_solver.py`
- Ensure `mnist_digit_classifier.keras` exists in the project directory

**Port already in use:**
- Change the port in `app.py` (line 48):
  ```python
  app.run(debug=True, port=5001)  # Use different port
  ```

**Poor predictions:**
- Try drawing larger, clearer digits
- Ensure the digit is centered in the canvas
- Avoid complex or stylized handwriting

## License

This project uses the MNIST dataset which is freely available for research and educational purposes.
