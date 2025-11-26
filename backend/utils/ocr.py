# backend/utils/ocr.py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import fitz  # PyMuPDF
import os
import re

# ——— Load Florence-2 (Lighter, Faster, Crash-Proof) ———
print("⏳ Loading Florence-2 (OCR Engine)...")

# Note: We don't need bitsandbytes quantization here because 
# Florence-2 is small enough (0.7B) to fit in your 4GB VRAM natively!
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", 
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

print(f"✅ Florence-2 Ready on {device.upper()}! (No quantization needed)\n")

def extract_text_from_file(file_path: str) -> str:
    """Accepts PDF or image path -> returns clean extracted text"""
    if file_path.lower().endswith(".pdf"):
        try:
            # 1. Try to extract text directly (fastest)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # 2. If empty, use AI Vision (OCR)
            if not text.strip():
                print("   (PDF is scanned image, running AI Vision...)")
                return _ocr_image_from_pdf(file_path)
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {e}"
    else:  # It's an image
        return _ocr_image(file_path)

def _ocr_image(image_input) -> str:
    """Helper: Runs Florence-2 Vision on an image."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # Florence-2 uses specific prompts. '<OCR>' tells it to read text.
    prompt = "<OCR>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    
    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process: Extract the text from the result
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(image.width, image.height)
    )
    
    return parsed_answer.get(prompt, "")

def _ocr_image_from_pdf(pdf_path: str) -> str:
    """Fallback: convert first page of Scanned PDF to image and OCR"""
    doc = fitz.open(pdf_path)
    page = doc[0] # Analyze first page
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return _ocr_image(img)

def detect_missing_chars(text: str) -> dict:
    """Analyze text and return what the user needs to write"""
    text = text.lower()
    
    # 1. Letters
    all_letters = set("abcdefghijklmnopqrstuvwxyz")
    present_letters = set(re.findall(r"[a-z]", text))
    missing_letters = all_letters - present_letters
    
    # 2. Numbers & symbols
    all_symbols = set("0123456789@#$%&*()_+-=[]{}|;':\",.<>?/\\")
    present_symbols = set(re.findall(r"[0-9@#$%&*()_+\-=[\]{}|;':\",.<>?/\\]", text))
    missing_symbols = all_symbols - present_symbols
    
    # 3. Common words
    common_words = {"the", "of", "and", "to", "in", "i", "that", "was", "his", "he"}
    words_in_text = set(re.findall(r"\b[a-z]+\b", text))
    missing_words = common_words - words_in_text
    
    return {
        "missing_letters": sorted(list(missing_letters)),
        "missing_symbols": sorted(list(missing_symbols)),
        "missing_common_words": sorted(list(missing_words)),
        "full_text": text[:500] + ("..." if len(text) > 500 else "")
    }

# ——— Quick Test ———
if __name__ == "__main__":
    test_file = "data/test_docs/test_image.png" 
    
    print(f"🔎 Looking for file: {test_file}")
    if os.path.exists(test_file):
        print(f"🚀 Processing...")
        text = extract_text_from_file(test_file)
        result = detect_missing_chars(text)
        
        print("\n——— RESULTS ———")
        print(f"📝 Text Found: {result['full_text']}\n")
        print(f"❌ Missing Letters: {result['missing_letters']}")
    else:
        print(f"⚠️ Test file not found at: {test_file}")
        print("Please put an image or PDF in 'HandwritingAI/data/test_docs/' and name it 'test_image.png'")