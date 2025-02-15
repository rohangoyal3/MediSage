# Medisage - AI-Powered Prescription Decoder

### Description
Medisage is a web application that leverages AI-powered Optical Character Recognition (OCR) and Natural Language Processing (NLP) to help patients understand handwritten prescriptions. By extracting and deciphering key prescription details—such as medicine names, dosages, and usage instructions—Medisage provides clear explanations and suggests verified alternatives if a prescribed drug is unavailable. This innovative solution addresses common challenges like illegible handwriting, lack of medication awareness, and risks of medical errors, ultimately improving medication awareness, reducing errors, and enhancing healthcare accessibility.

### Project Workflow
1. __Image Processing & OCR__
    * Load prescription image
    * Preprocess (grayscale, thresholding, noise removal)
    * Extract text using Tesseract OCR / EasyOCR
    * Clean and structure extracted text

2. __Medicine Name Matching__
    * Convert dataset medicine names into a searchable format
    * Implement fuzzy string matching for medicine identification
    * Handle OCR spelling variations and recognition errors

3. __Backend API Development__
    * Set up FastAPI for handling medicine lookup requests
    * Implement API to process extracted medicine names
    * Query PostgreSQL (or SQLite) database for medicine details
    * Return:
      * Medicine details (name, price, manufacturer, type, composition)
      * Alternative medicines if unavailable/discontinued
    * Optimize for fast queries and API response times

4. __Frontend UI Development__
    * Design React.js UI with Tailwind CSS
    * Implement File Upload feature for prescription images
    * Connect UI to Backend API for medicine lookup
    * Display:
      * Extracted prescription text
      * Medicine details & alternative suggestions

5. __Integration & Testing__
    * Connect OCR pipeline with Backend API
    * Ensure correct data flow between frontend and backend
    * Test with multiple handwritten prescriptions
    * Debug UI issues, API errors, and database queries

6. __Deployment & Hackathon Demo Preparation__
    * Deploy Backend API on Render/Railway
    * Deploy Frontend on Vercel
    * Prepare real-world test cases for demo
    * Finalize presentation showcasing problem, solution, and impact

### Dataset Used
* Indian Medicine Dataset
* Doctor's Handwritten Prescription BD Dataset

### Tech Stack
__Frontend:__
* React.js
* Tailwind CSS
  
__Backend:__
* FastAPI
* Uvicorn

__OCR & Image Processing:__
* OpenCV
* PyTesseract
* EasyOCR
* Pillow

__NLP for Medical Advice:__
* Hugging Face Transformers
* BioBERT (dmis-lab/biobert-base-cased)
  
__Data Processing:__
* NumPy
* Pandas

__Deployment:__
* Frontend → Vercel
* Backend → Render / Railway
* NLP Model → Hugging Face Inference API

### Expected Deliverables
* Fully functional web application for prescription decoding
* AI-powered OCR to extract and structure prescription text
* Fuzzy string matching for medicine identification
* Medicine lookup API with alternatives and verified drug information
* Intuitive frontend UI for user interaction
* Optimized deployment for seamless access and fast responses
* Comprehensive testing with real-world handwritten prescriptions

Medisage aims to bridge the gap between handwritten prescriptions and accessible healthcare information, ensuring patients receive clear and accurate medication details while enhancing overall prescription management.
