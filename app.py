from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
print("Loading MNIST model...")
model = keras.models.load_model('mnist_digit_classifier.keras')
print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from request
        data = request.json.get('image')

        if not data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove the data URL prefix
        image_data = data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale and resize to 28x28
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0

        # Invert if needed (MNIST has white digits on black background)
        if np.mean(img_array) > 0.5:
            img_array = 1 - img_array

        # Reshape for prediction (1, 784)
        img_flat = img_array.reshape(1, 784)

        # Make prediction
        prediction = model.predict(img_flat, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {'digit': int(idx), 'probability': float(prediction[0][idx])}
            for idx in top_3_indices
        ]

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'top_predictions': top_3_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
