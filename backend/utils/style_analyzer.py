# backend/utils/style_analyzer.py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os

# ——— Load Florence-2 (Using the Microsoft version that works for you) ———
print("⏳ Loading Style Analyzer...")

# We use the exact same ID as your working OCR module to reuse the cache
MODEL_ID = "microsoft/Florence-2-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch_dtype, 
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)

# Load Processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("✅ Style Analyzer Ready!\n")

def analyze_handwriting_style(image_path):
    """
    Asks Florence-2 to describe the visual style of the handwriting.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"full_description": "Error loading image", "detected_style_tags": []}

    # Task: Describe the image in detail
    prompt = "<MORE_DETAILED_CAPTION>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse the result
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(image.width, image.height)
    )
    
    description = parsed_answer.get(prompt, "")
    
    # Basic Style Tagging (We will make this smarter later)
    style_tags = []
    desc_lower = description.lower()
    
    if "cursive" in desc_lower: style_tags.append("Cursive")
    if "print" in desc_lower or "block" in desc_lower: style_tags.append("Print")
    if "messy" in desc_lower or "scribble" in desc_lower: style_tags.append("Messy")
    if "neat" in desc_lower: style_tags.append("Neat")
    if "slant" in desc_lower: style_tags.append("Slanted")
    if "handwritten" in desc_lower or "handwriting" in desc_lower: style_tags.append("Handwritten")

    return {
        "full_description": description,
        "detected_style_tags": style_tags
    }

# ——— Test It ———
if __name__ == "__main__":
    # Use the same test image
    test_file = "data/test_docs/test_image.jpg" 
    
    if os.path.exists(test_file):
        print(f"🎨 Analyzing style of: {test_file}")
        result = analyze_handwriting_style(test_file)
        
        print("\n——— STYLE REPORT ———")
        print(f"📝 AI Description: {result['full_description']}")
        print(f"🏷️  Tags: {result['detected_style_tags']}")
    else:
        print(f"⚠️ File not found: {test_file}")