from fastapi import FastAPI, File, UploadFile
import shutil
import os
from scripts.ocr import process_uploaded_image
from scripts.search import search_names_in_xlsx

app = FastAPI()

UPLOAD_DIR = "data/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-prescription/")
async def upload_prescription(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image using OCR and extract text
    extracted_text = process_uploaded_image(file_path)
    extracted_details = search_names_in_xlsx(extracted_text)

    return {"message": "File processed successfully", "filename": file.filename, "extracted_text": extracted_text,"Details": extracted_details}
