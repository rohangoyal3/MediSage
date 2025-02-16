import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import re

# 🔹 Replace with your actual Gemini API Key
API_KEY = "AIzaSyDRQJYsHRpRzTTMmGiCMBLE4FK1M9qibJg"

# Configure Gemini API
genai.configure(api_key=API_KEY)

def preprocess_image(image_path):
    """ Preprocess image: Crop text, denoise, resize for better OCR. """
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection to find text regions
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours (text bounding areas)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest bounding box (likely text area)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Expand bounding box with padding
    padding = 20
    x, y, w, h = max(0, x - padding), max(0, y - padding), w + 2 * padding, h + 2 * padding

    # Crop the image to focus on text
    cropped = gray[y:y+h, x:x+w]

    # Resize image for better OCR accuracy
    resized = cv2.resize(cropped, (640, 480), interpolation=cv2.INTER_CUBIC)

    return resized

def extract_text_gemini(image_path):
    """ Convert preprocessed image to bytes and send to Gemini for OCR """

    # Preprocess the image (returns a NumPy array)
    processed_img = preprocess_image(image_path)

    # Convert to PIL Image (for compatibility)
    pil_image = Image.fromarray(processed_img)

    # Convert PIL Image to byte stream (Gemini requires bytes)
    img_byte_array = io.BytesIO()
    pil_image.save(img_byte_array, format="PNG")  # ✅ Convert image to PNG format
    img_bytes = img_byte_array.getvalue()

    # Load Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ✅ Send image as a blob (correct format)
    response = model.generate_content(
        [
            {"mime_type": "image/png", "data": img_bytes},  # ✅ Correct blob format
            "Extract only the words exactly as they appear in this image, without adding any extra text or explanations."
        ]
    )

    return response.text if response else "No text extracted."

# Example Usage
image_path = "img.jpg"  # Change to your image path
text = extract_text_gemini(image_path)

words = re.findall(r'\b\w+\b', text)  # Extracts all words
print(words)

