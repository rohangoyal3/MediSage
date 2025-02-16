import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import re

# Configure Gemini API
API_KEY = "AIzaSyAAoh22e0DDMTLw1q3J8DxHEPCUcstAplQ"
genai.configure(api_key=API_KEY)


def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    padding = 20
    x, y, w, h = max(0, x - padding), max(0, y - padding), w + 2 * padding, h + 2 * padding

    cropped = gray[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (640, 480), interpolation=cv2.INTER_CUBIC)

    return resized


def extract_text_from_image(image_path):
    """Extracts text from an image using Gemini AI."""
    processed_img = preprocess_image(image_path)
    pil_image = Image.fromarray(processed_img)

    img_byte_array = io.BytesIO()
    pil_image.save(img_byte_array, format="PNG")
    img_bytes = img_byte_array.getvalue()

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            {"mime_type": "image/png", "data": img_bytes},
            "Extract only the words exactly as they appear in this image. Return the words **inside double quotes** and separated by a space, like this: 'word1' 'word2' 'word3' Do **not** add any extra text, explanations, or formatting."
        ]
    )

    return re.findall(r"'(.*?)'", response.text) if response else []

# Function to process uploaded image
def process_uploaded_image(file_path):
    return extract_text_from_image(file_path)