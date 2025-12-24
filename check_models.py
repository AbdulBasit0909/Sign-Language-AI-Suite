import google.generativeai as genai

# PASTE YOUR KEY HERE
GEMINI_API_KEY = "AIzaSyDPbWimBRZ2z_scNctNK1tcG2qTeofG-1o" 

genai.configure(api_key=GEMINI_API_KEY)

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Found: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")