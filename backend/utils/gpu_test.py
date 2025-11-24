# gpu_test.py - Your First Learning Module

# === PART 1: Import Libraries ===
# Think of imports like calling tools from your toolbox
import torch  # PyTorch - the "brain" of AI, handles math on GPU

print("=== 🎓 GPU Learning Test ===\n")
print("Let's check if your GTX 1650 is ready!\n")

# === PART 2: Check Basic GPU Info ===
# This is like asking your computer "Do you have a graphics card?"
print("📊 Step 1: Checking PyTorch Installation...")
print(f"   PyTorch Version: {torch.__version__}")

# The magic question: Can PyTorch "see" your NVIDIA GPU?
cuda_available = torch.cuda.is_available()
print(f"   CUDA Available: {cuda_available}")

if cuda_available:
    # If yes, get details
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    
    # How much VRAM (video memory) do you have?
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert bytes to GB
    print(f"   Total VRAM: {total_vram:.2f} GB")
    print(f"   VRAM Currently Used: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB\n")
    
    print("✅ SUCCESS! Your GPU is detected.\n")
else:
    print("❌ CUDA not detected. Your GPU isn't being used.")
    print("   Fix: Update NVIDIA drivers from nvidia.com\n")
    exit()  # Stop here if GPU not working

# === PART 3: Test Quantization (The Magic Trick) ===
# This is THE KEY to running big AI models on your 4GB GPU
print("📊 Step 2: Testing 4-bit Quantization...")
print("   (This compresses models to fit in your VRAM)\n")

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    # Configure quantization - these settings compress models by 75%!
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit precision (vs normal 16-bit)
        bnb_4bit_compute_dtype=torch.float16,  # Still compute in float16 for accuracy
        bnb_4bit_quant_type="nf4",  # "NormalFloat4" - best quality compression
        bnb_4bit_use_double_quant=True,  # Extra compression layer
    )
    # Load a tiny test model (125 million parameters)
    print("   Loading test model (facebook/opt-125m)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", 
        quantization_config=quant_config,
        device_map="auto",
        use_safetensors=True  # <--- THIS IS THE FIX (Forces secure format)
    )
    
    # Check memory usage
    vram_used = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"\n✅ Quantization WORKS!")
    print(f"   Model loaded successfully")
    print(f"   VRAM Used: {vram_used:.2f} GB")
    print(f"   💡 Normal 16-bit would use ~0.5GB, we're using ~{vram_used:.2f}GB")
    print(f"   That's {((0.5 - vram_used) / 0.5 * 100):.0f}% savings!\n")
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Quantization test failed!")
    print(f"   Error: {e}\n")
    print("   This means bitsandbytes isn't working. Let me know this error!")

# === PART 4: Summary ===
print("=== 🎉 Test Complete! ===\n")
print("What you just learned:")
print("1. How to check if GPU is detected (torch.cuda.is_available)")
print("2. How to see VRAM usage (important for monitoring)")
print("3. How quantization compresses models to fit in 4GB VRAM")
print("\nIf you see ✅ above, you're ready to build the OCR module!")
print("Next: We'll load PaliGemma (the real AI for handwriting)")
