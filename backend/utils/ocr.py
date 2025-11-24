# backend/utils/ocr.py
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import fitz  # PyMuPDF
import os
import re

# ——— 4-bit Quantization Config (The "Compressor") ———
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ——— Load PaliGemma (The "Brain") ———
print("⏳ Loading PaliGemma... (This takes ~2-5 mins the first time)")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,   # <--- CRITICAL: Prevents RAM Crash!
    use_safetensors=True      # <--- CRITICAL: Fixes Security Error!
)
print("✅ PaliGemma Ready! VRAM ≈ 2.2 GB\n")

def extract_text_from_file(file_path: str) -> str:
    """Accepts PDF or image path -> returns clean extracted text"""
    if file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            # If PDF has text layers, return them. If empty, use OCR (scanned PDF).
            return text.strip() if text.strip() else _ocr_image_from_pdf(file_path)
        except Exception as e:
            return f"Error reading PDF: {e}"
    else:  # It's an image
        return _ocr_image(file_path)

def _ocr_image(image_input) -> str:
    """
    Helper: Runs AI Vision on an image.
    Handles both file paths (str) and PIL Image objects.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB") # It's already an object from PDF function

    # The Prompt: We tell the AI what to look for
    prompt = "extract text"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
    # Generate text (limit to 512 tokens to save memory)
    output = model.generate(**inputs, max_new_tokens=512)
    
    # Decode result
    text = processor.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the result if the model repeats it
    return text.replace(prompt, "").strip()

def _ocr_image_from_pdf(pdf_path: str) -> str:
    """Fallback: convert first page of Scanned PDF to image and OCR"""
    doc = fitz.open(pdf_path)
    page = doc[0]
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
    # Ensure this path exists or change it!
    test_file = "data/test_docs/test_image.png" 
    
    print(f"🔎 Looking for file: {test_file}")
    if os.path.exists(test_file):
        print(f"🚀 Processing...")
        text = extract_text_from_file(test_file)
        result = detect_missing_chars(text)
        
        print("\n——— RESULTS ———")
        print(f"📝 Text Found: {result['full_text']}\n")
        print(f"❌ Missing Letters: {result['missing_letters']}")
        print(f"❌ Missing Symbols: {result['missing_symbols']}")
    else:
        print(f"⚠️ Test file not found at: {test_file}")
        print("Please put an image or PDF in 'HandwritingAI/data/test_docs/' and name it 'test_image.png'")