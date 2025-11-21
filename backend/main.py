from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import pdfplumber
import pytesseract
import os
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Point pytesseract at your Tesseract install on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Handwriting AI backend is up and running!"}

@app.get("/echo")
def echo(text: str):
    return {"echoed_text": text}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = ["jpg", "jpeg", "png", "pdf"]
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. Please upload an image (jpg, jpeg, png) or pdf."
        )

    # Ensure 'data/raw' exists
    os.makedirs("data/raw", exist_ok=True)
    file_location = f"data/raw/{filename}"

    # Save uploaded file to disk
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    extracted_text = None
    corrected_text = None

    # OCR/image extraction
    if ext in ["jpg", "jpeg", "png"]:
        try:
            extracted_text = extract_text_from_image(file_location)
        except UnidentifiedImageError:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Uploaded image file is not a valid image.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image OCR error: {str(e)}")
    elif ext == "pdf":
        try:
            extracted_text = extract_text_from_pdf(file_location)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")

    # Grammar and spelling correction
    if extracted_text and extracted_text.strip():
        try:
            corrected_text = correct_text(extracted_text)
        except Exception as e:
            corrected_text = None

    # Plagiarism detection (dummy references for demo)
    reference_texts = [
        "This is a sample reference article or essay for plagiarism checking.",
        "Another example document to compare against."
    ]
    plagiarism_score = 0.0
    if corrected_text:
        plagiarism_score = check_plagiarism(corrected_text, reference_texts)

    return {
        "info": f"file '{filename}' saved at '{file_location}'",
        "type": ext,
        "extracted_text_sample": extracted_text[:200] if extracted_text else "No text found",
        "corrected_text_sample": corrected_text[:200] if corrected_text else "Nothing to correct or error",
        "plagiarism_score": plagiarism_score  # Value from 0.0 (unique) to 1.0 (copied)
    }

from fastapi import Form

@app.post("/uploadsample/")
async def upload_handwriting_sample(label: str = Form(...), file: UploadFile = File(...)):
    allowed_extensions = ["jpg", "jpeg", "png"]
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only image files (.jpg, .jpeg, .png) are accepted for handwriting samples."
        )

    # Ensure 'data/user_samples' exists
    os.makedirs("data/user_samples", exist_ok=True)
    # Save as: data/user_samples/a.png or hello.png, etc
    safe_label = "".join(c for c in label if c.isalnum() or c in ('_', '-')).rstrip()
    file_location = f"data/user_samples/{safe_label}.{ext}"
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save handwriting sample: {str(e)}")
    return {"info": f"Handwriting sample for '{label}' saved as '{file_location}'"}

# --------------------------
# Helper functions below
# --------------------------

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def correct_text(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected

def check_plagiarism(candidate_text, reference_texts):
    docs = [candidate_text] + reference_texts
    tfidf = TfidfVectorizer().fit_transform(docs)
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    highest = max(similarities) if len(similarities) > 0 else 0
    return highest
