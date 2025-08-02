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

# Available models to test (image-compatible models only)
AVAILABLE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]


def select_model():
    """Allow user to select a model to test"""
    print("\n--- Available Models for Testing ---")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model}")

    while True:
        try:
            choice = input(
                f"\nSelect a model (1-{len(AVAILABLE_MODELS)}) "
                "or 'q' to quit: "
            ).strip()

            if choice.lower() == "q":
                return None

            choice = int(choice)

            if 1 <= choice <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


def test_model(model_name):
    """Test a specific model with text and image generation"""
    print(f"\n{'=' * 50}")
    print(f"Testing model: {model_name}")
    print(f"{'=' * 50}")

    try:
        # Test text generation
        print("\n--- Testing Text Generation ---")
        text_model = genai.GenerativeModel(model_name)
        text_prompt = "Write a short, happy greeting."

        try:
            text_response = text_model.generate_content(text_prompt)

            if text_response.text:
                print("✅ Success! Text generation works.")
                print(f"Generated text: {text_response.text.strip()}")
            else:
                print("❌ Text generation returned no content.")
                if hasattr(text_response, "prompt_feedback"):
                    print(f"Prompt feedback: {text_response.prompt_feedback}")
        except Exception as e:
            print(f"❌ Text generation failed: {e}")

        # Test image understanding (only for vision-capable models)
        print("\n--- Testing Image Understanding ---")
        try:
            # Create a simple test image
            img = Image.new("RGB", (100, 100), color="blue")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            vision_prompt = "Describe this image briefly."
            vision_model = genai.GenerativeModel(model_name)
            vision_response = vision_model.generate_content([
                vision_prompt,
                img,
            ])

            if vision_response.text:
                print("✅ Success! Image understanding works.")
                print(
                    f"Generated description: {vision_response.text.strip()}"
                )
            else:
                print("❌ Image understanding returned no content.")
                if hasattr(vision_response, "prompt_feedback"):
                    print(
                        f"Prompt feedback: {vision_response.prompt_feedback}"
                    )
        except Exception as e:
            print(f"❌ Image understanding failed: {e}")
            if (
                "vision" not in model_name.lower()
                and "flash" not in model_name.lower()
            ):
                print("   (This model might not support image inputs)")

    except Exception as e:
        print(f"❌ Failed to initialize model {model_name}: {e}")


if __name__ == "__main__":
    print("Google Gemini API Model Tester")
    print("==============================")

    while True:
        selected_model = select_model()
        if selected_model is None:
            print("Goodbye!")
            break

        # Test the selected model
        test_model(selected_model)

        # Ask if user wants to test another model
        while True:
            another = input("\nTest another model? (y/n): ").strip().lower()
            if another in ["y", "yes"]:
                break
            elif another in ["n", "no"]:
                print("Goodbye!")
                exit()
            else:
                print("Please enter 'y' or 'n'.")
