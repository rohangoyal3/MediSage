import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ✅ Define paths
TRAIN_LABELS_CSV_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/train/training_labels.csv"
TRAIN_IMAGES_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/train/training_words/"
MODEL_SAVE_PATH = "C:/Users/mohit/PycharmProjects/MediSage/models/handwritten_ocr_model.h5"

# ✅ Load training labels CSV
train_labels_df = pd.read_csv(TRAIN_LABELS_CSV_PATH)

# ✅ Convert CSV to dictionary: { "0.png": "Aceta", "1.png": "Ibuprofen", ... }
train_labels_dict = dict(zip(train_labels_df["IMAGE"], train_labels_df["MEDICINE_NAME"]))

print(f"✅ Loaded {len(train_labels_dict)} training samples.")

# ✅ Tokenizer for encoding text labels
medicine_names = train_labels_df["MEDICINE_NAME"].astype(str).tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(medicine_names)

# ✅ Convert medicine names to numerical sequences
train_labels_encoded = tokenizer.texts_to_sequences(medicine_names)

# ✅ Define a fixed max length for sequences
MAX_SEQ_LENGTH = max(len(seq) for seq in train_labels_encoded)

# ✅ Pad all sequences to the same length
train_labels_encoded = pad_sequences(train_labels_encoded, maxlen=MAX_SEQ_LENGTH, padding="post")

# ✅ Convert labels to one-hot encoding (No reshaping required)
NUM_CLASSES = len(tokenizer.word_index) + 1  # Get actual number of classes
train_labels_encoded = to_categorical(train_labels_encoded, num_classes=NUM_CLASSES)

print("✅ Labels successfully converted to one-hot encoding!")
print(f"✅ Label Shape After Fix: {train_labels_encoded.shape}")  # Should be (None, MAX_SEQ_LENGTH, NUM_CLASSES)


# ✅ Function to preprocess images for CNN input
def preprocess_image(image_path):
    """Convert image to grayscale, resize, normalize, and reshape for CNN."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Error: Cannot read image at {image_path}")
        return None

    img = cv2.resize(img, (128, 32))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (for CNN)
    return img


# ✅ Load images and preprocess them
train_images = []
valid_images = []
train_labels = []
valid_labels = []

# Split data: 90% training, 10% validation
split_ratio = 0.9
split_index = int(len(train_labels_dict) * split_ratio)

for i, (filename, label) in enumerate(train_labels_dict.items()):
    image_path = os.path.join(TRAIN_IMAGES_PATH, filename)
    if os.path.exists(image_path):
        img = preprocess_image(image_path)
        if img is not None:
            if i < split_index:
                train_images.append(img)
                train_labels.append(train_labels_encoded[i])  # Use correct one-hot labels
            else:
                valid_images.append(img)
                valid_labels.append(train_labels_encoded[i])  # Use correct one-hot labels

train_images = np.array(train_images)
valid_images = np.array(valid_images)
train_labels_encoded = np.array(train_labels)
valid_labels_encoded = np.array(valid_labels)

print(f"✅ Final Training Samples: {len(train_images)} images.")
print(f"✅ Final Validation Samples: {len(valid_images)} images.")

# ✅ Define CNN + LSTM Model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Reshape((-1, 128)),  # ✅ Fix: Automatically adjust based on input size
    LSTM(64, return_sequences=True),
    LSTM(64),

    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')  # ✅ Fix: Match model output to number of classes
])

# ✅ Print model input shape
print("Model Input Shape:", model.input_shape)

# ✅ Compile model (Fix: Use categorical_crossentropy)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Show model summary
model.summary()

# ✅ Train the model
history = model.fit(
    train_images, train_labels_encoded,
    epochs=20,
    batch_size=32,
    validation_data=(valid_images, valid_labels_encoded)
)

# ✅ Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete and saved at {MODEL_SAVE_PATH}!")
