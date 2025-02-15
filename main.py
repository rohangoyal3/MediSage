from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads/"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-prescription/")
async def upload_prescription(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "File uploaded successfully", "filename": file.filename, "file_path": file_path}
