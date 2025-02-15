import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Define paths
MODEL_PATH = "C:/Users/mohit/PycharmProjects/MediSage/models/handwritten_ocr_model.h5"
LABELS_CSV_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/train/training_labels.csv"
IMAGES_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/test/testing_words/"

# Load the trained OCR model
model = load_model(MODEL_PATH)

# Load labels to get tokenizer vocabulary
labels_df = pd.read_csv(LABELS_CSV_PATH)
medicine_names = labels_df["MEDICINE_NAME"].astype(str).tolist()

# Tokenizer for decoding predictions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(medicine_names)


def preprocess_image(image_path):
    """Convert image to grayscale, resize, normalize, and reshape for CNN."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"❌ Error: Image file not found at {image_path}")

    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img


def predict_text(image_path):
    """Predict medicine name from handwritten image using trained CNN+LSTM model."""
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    decoded_text = tokenizer.sequences_to_texts([np.argmax(prediction, axis=1)])

    return decoded_text[0]


if __name__ == "__main__":
    image_path = os.path.join(IMAGES_PATH, "0.png")  # Example test image
    try:
        predicted_text = predict_text(image_path)
        print(f"🔍 Predicted Medicine Name: {predicted_text}")
    except Exception as e:
        print(f"❌ Error: {e}")
