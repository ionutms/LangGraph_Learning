# test_gemini_api.py
import io
import os

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)
else:
    print("API Key loaded successfully.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash-lite"


if __name__ == "__main__":
    print(f"Testing access to Google Gemini model: {MODEL_NAME}")

    print("\n--- Testing Text Generation ---")
    text_model = genai.GenerativeModel(MODEL_NAME)
    text_prompt = "Write a short, happy greeting."
    text_response = text_model.generate_content(text_prompt)

    if text_response.text:
        print("Success! Text generation works.")
        print(f"Generated text: {text_response.text.strip()}")
    else:
        print("Text generation failed or returned no content.")
        print(f"Response: {text_response}")

    print("\n--- Testing Image Understanding ---")
    img = Image.new("RGB", (100, 100), color="blue")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    vision_prompt = "Describe this image briefly."
    vision_model = genai.GenerativeModel(MODEL_NAME)
    vision_response = vision_model.generate_content([vision_prompt, img])

    if vision_response.text:
        print("Success! Image understanding works.")
        print(f"Generated description: {vision_response.text.strip()}")
    else:
        print("Image understanding failed or returned no content.")
        print(f"Response: {vision_response}")
