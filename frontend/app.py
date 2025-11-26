import streamlit as st
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.ocr import extract_text_from_file, detect_missing_chars
from backend.utils.style_analyzer import analyze_handwriting_style
from backend.utils.suggester import create_smart_instruction
from backend.utils.generator import generate_handwriting_image

st.set_page_config(page_title="Handwriting AI", page_icon="✍️", layout="wide")

st.title("✍️ Your Handwriting AI")
st.markdown("### Turn digital text into *your* unique handwriting.")

if "step" not in st.session_state: st.session_state.step = 1
if "style_desc" not in st.session_state: st.session_state.style_desc = "Handwritten text"

# ——— COLUMNS ———
col1, col2 = st.columns([1, 1])

# ——— LEFT: DATA GATHERING ———
with col1:
    st.header("1. Content & Data")
    uploaded_doc = st.file_uploader("Upload Source Document", type=["pdf", "png", "jpg"], key="doc")

    if uploaded_doc:
        path = f"data/test_docs/{uploaded_doc.name}"
        os.makedirs("data/test_docs", exist_ok=True)
        with open(path, "wb") as f: f.write(uploaded_doc.read())
        
        if st.button("🔍 Analyze Text"):
            with st.spinner("Reading..."):
                text = extract_text_from_file(path)
                missing = detect_missing_chars(text)
                st.session_state.doc_text = text
                st.session_state.missing = missing
                st.session_state.step = 2

    if "missing" in st.session_state and st.session_state.missing:
        st.divider()
        st.text_area("Extracted Text", st.session_state.doc_text, height=100)
        
        # Smart Instruction Display
        miss_data = st.session_state.missing
        if miss_data["missing_letters"]:
            st.info(create_smart_instruction(miss_data["missing_letters"], []))
        else:
            st.success("✅ All letters present! Just upload a style sample.")

# ——— RIGHT: STYLE & GENERATE ———
with col2:
    if st.session_state.step >= 2:
        st.header("2. Style & Generate")
        uploaded_style = st.file_uploader("Upload Handwriting Sample", type=["png", "jpg"], key="style")
        
        if uploaded_style:
            path = "data/user_samples/style.jpg"
            os.makedirs("data/user_samples", exist_ok=True)
            with open(path, "wb") as f: f.write(uploaded_style.read())
            st.image(uploaded_style, width=200)
            
            if st.button("🎨 Learn Style"):
                with st.spinner("Analyzing vibe..."):
                    report = analyze_handwriting_style(path)
                    st.session_state.style_desc = report['full_description']
                    st.session_state.step = 3
                    st.success("Style Learned!")
                    st.caption(report['full_description'])

    # ——— GENERATION BUTTON ———
    if st.session_state.step == 3:
        st.divider()
        st.header("3. Final Output")
        
        # Text input for custom generation (or use extraction)
        final_text = st.text_area("Text to Write:", st.session_state.doc_text, height=100)
        
        if st.button("🚀 Generate Handwriting", type="primary"):
            status = st.empty()
            status.info("⏳ Loading Flux Engine (This takes time on first run)...")
            
            # Call the Generator
            try:
                image = generate_handwriting_image(final_text[:200], st.session_state.style_desc) # Limit len for demo
                
                status.success("Generated!")
                st.image(image, caption="AI Generated Handwriting", use_column_width=True)
                
                # Save
                out_path = "data/outputs/final_result.png"
                image.save(out_path)
                with open(out_path, "rb") as file:
                    st.download_button("Download Image", file, "handwriting.png", "image/png")
            except Exception as e:
                status.error(f"Error: {e}")