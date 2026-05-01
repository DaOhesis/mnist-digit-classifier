import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import sys

# Load the trained model
print("Loading trained model...")
model = keras.models.load_model('mnist_digit_classifier.keras')
print("Model loaded successfully!")

# Load MNIST test data for demonstration
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test_flat = x_test.reshape((x_test.shape[0], 784))

def predict_digit(image_index):
    """Predict the digit for a given test image index"""
    image = x_test[image_index]
    true_label = y_test[image_index]

    # Reshape for prediction
    image_flat = image.reshape(1, 784)

    # Make prediction
    prediction = model.predict(image_flat, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f'True: {true_label} | Predicted: {predicted_class} | Confidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()

    # Show probability distribution
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), prediction[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Prediction Probability Distribution')
    plt.xticks(range(10))
    plt.show()

    return predicted_class, confidence

def predict_custom_image(image_path):
    """Predict digit from a custom image file"""
    from PIL import Image

    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0

        # Invert if needed (MNIST has white digits on black background)
        if np.mean(img_array) > 0.5:
            img_array = 1 - img_array

        # Reshape for prediction
        img_flat = img_array.reshape(1, 784)

        # Make prediction
        prediction = model.predict(img_flat, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display results
        plt.figure(figsize=(6, 4))
        plt.imshow(img_array, cmap='gray')
        plt.title(f'Predicted: {predicted_class} | Confidence: {confidence:.2%}')
        plt.axis('off')
        plt.show()

        # Show probability distribution
        plt.figure(figsize=(10, 4))
        plt.bar(range(10), prediction[0])
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Prediction Probability Distribution')
        plt.xticks(range(10))
        plt.show()

        return predicted_class, confidence

    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

# Interactive prediction
print("\n" + "="*60)
print("MNIST Digit Classifier - Interactive Prediction")
print("="*60)

while True:
    print("\nOptions:")
    print("1. Predict from test dataset (enter image index 0-9999)")
    print("2. Predict from custom image file")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        try:
            idx = int(input("Enter image index (0-9999): "))
            if 0 <= idx < len(x_test):
                predict_digit(idx)
            else:
                print("Index out of range!")
        except ValueError:
            print("Please enter a valid number!")

    elif choice == '2':
        path = input("Enter image path: ").strip()
        predict_custom_image(path)

    elif choice == '3':
        print("Goodbye!")
        break

    else:
        print("Invalid choice!")
