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

TRAIN_LABELS_CSV_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/train/training_labels.csv"
TRAIN_IMAGES_PATH = "C:/Users/mohit/PycharmProjects/MediSage/data/prescription_dataset/train/training_words/"
MODEL_SAVE_PATH = "C:/Users/mohit/PycharmProjects/MediSage/models/handwritten_ocr_model.h5"

train_labels_df = pd.read_csv(TRAIN_LABELS_CSV_PATH)

train_labels_dict = dict(zip(train_labels_df["IMAGE"], train_labels_df["MEDICINE_NAME"]))

print(f"✅ Loaded {len(train_labels_dict)} training samples.")

medicine_names = train_labels_df["MEDICINE_NAME"].astype(str).tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(medicine_names)

train_labels_encoded = tokenizer.texts_to_sequences(medicine_names)

NUM_CLASSES = len(tokenizer.word_index) + 1
train_labels_encoded = [to_categorical(seq, num_classes=NUM_CLASSES) for seq in train_labels_encoded]

MAX_SEQ_LENGTH = max(len(seq) for seq in train_labels_encoded)
train_labels_encoded = pad_sequences(train_labels_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", dtype='float32')

train_labels_encoded = np.array(train_labels_encoded).reshape(len(train_labels_encoded), -1)

print(f"✅ Fixed Label Shape: {train_labels_encoded.shape}")  # Should be (None, NUM_CLASSES)


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Error: Cannot read image at {image_path}")
        return None

    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


train_images = []
valid_images = []
train_labels = []
valid_labels = []

split_ratio = 0.9
split_index = int(len(train_labels_dict) * split_ratio)

for i, (filename, label) in enumerate(train_labels_dict.items()):
    image_path = os.path.join(TRAIN_IMAGES_PATH, filename)
    if os.path.exists(image_path):
        img = preprocess_image(image_path)
        if img is not None:
            if i < split_index:
                train_images.append(img)
                train_labels.append(train_labels_encoded[i])
            else:
                valid_images.append(img)
                valid_labels.append(train_labels_encoded[i])

train_images = np.array(train_images)
valid_images = np.array(valid_images)
train_labels_encoded = np.array(train_labels)
valid_labels_encoded = np.array(valid_labels)

print(f"✅ Final Training Samples: {len(train_images)} images.")
print(f"✅ Final Validation Samples: {len(valid_images)} images.")

model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Reshape((-1, 128)),
    LSTM(64, return_sequences=True),
    LSTM(64),

    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

print("Model Input Shape:", model.input_shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_images, train_labels_encoded,
    epochs=20,
    batch_size=32,
    validation_data=(valid_images, valid_labels_encoded)
)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete and saved at {MODEL_SAVE_PATH}!")
