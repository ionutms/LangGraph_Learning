import io
import os
from pathlib import Path

import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

# Available models to choose from (image-compatible models only)
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

# Directory setup
PDFS_PATH = "./pdf_rag_image/pdfs"
IMAGES_PATH = "./pdf_rag_image/extracted_images"
RESULTS_PATH = "./pdf_rag_image/results"

os.makedirs(PDFS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def select_model():
    """Allow user to select a Gemini model to use."""
    print("\n🤖 Available Gemini Models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")

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
                selected_model = AVAILABLE_MODELS[choice - 1]
                print(f"✅ Selected model: {selected_model}")
                return selected_model
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q'.")


def list_available_pdfs():
    """List all PDF files in the pdfs directory."""
    pdf_files = list(Path(PDFS_PATH).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {PDFS_PATH}")
        return []

    print("📚 Available PDF files:")
    for pdf_index, pdf_file in enumerate(pdf_files, 1):
        print(f"  {pdf_index}. {pdf_file.name}")
    return pdf_files


def list_extracted_images():
    """List all extracted images from previous PDF processing."""
    image_files = []

    # Look for PNG files in subdirectories
    for pdf_dir in Path(IMAGES_PATH).iterdir():
        if pdf_dir.is_dir():
            for img_file in pdf_dir.glob("*.png"):
                image_files.append(img_file)

    if not image_files:
        print(f"⚠️  No extracted images found in {IMAGES_PATH}")
        print("💡 Process a PDF first to extract images.")
        return []

    print("🖼️  Available extracted images:")
    for img_index, img_file in enumerate(image_files, 1):
        # Extract info from filename
        parts = img_file.stem.split("_")
        if len(parts) >= 4:
            page_num = parts[1]
            img_num = parts[3]
            print(
                f"  {img_index}. {img_file.parent.name} - "
                f"Page {page_num}, Image {img_num}"
            )
        else:
            print(f"  {img_index}. {img_file.name}")

    return image_files


def save_image_to_disk(
    image: Image.Image,
    pdf_name: str,
    page_num: int,
    img_index: int,
    img_type: str,
) -> str:
    """Save image to disk and return the file path."""
    pdf_dir = os.path.join(IMAGES_PATH, Path(pdf_name).stem)
    os.makedirs(pdf_dir, exist_ok=True)

    filename = f"page_{page_num}_img_{img_index}_{img_type}.png"
    filepath = os.path.join(pdf_dir, filename)

    if image.mode in ("RGBA", "LA", "P"):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        rgb_image.paste(
            image,
            mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None,
        )
        rgb_image.save(filepath, "PNG", optimize=True)
    else:
        image.save(filepath, "PNG", optimize=True)
    print(f"💾 Saved: {filename}")
    return filepath


def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file - embedded images only."""
    print(f"🔍 Extracting images from: {Path(pdf_path).name}")
    doc = fitz.open(pdf_path)
    images = []
    pdf_name = Path(pdf_path).name

    print("  📎 Looking for embedded images...")
    total_embedded = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        if image_list:
            print(
                f"    Page {page_num + 1}: "
                f"Found {len(image_list)} embedded images"
            )

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image and base_image.get("image"):
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # Only save reasonably sized images
                if image.width >= 50 and image.height >= 50:
                    image_path = save_image_to_disk(
                        image,
                        pdf_name,
                        page_num + 1,
                        img_index,
                        "embedded",
                    )

                    images.append({
                        "page": page_num + 1,
                        "index": img_index,
                        "image": image,
                        "pdf_name": pdf_name,
                        "image_path": image_path or "",
                        "type": "embedded",
                    })
                    total_embedded += 1
                    print(
                        f"🖼️ Extracted image {img_index} "
                        f"({image.width}x{image.height})"
                    )
                else:
                    print(
                        f"⚠️ Skipped small image {img_index} "
                        f"({image.width}x{image.height})"
                    )

    doc.close()
    print(f"  📊 Total embedded images found: {total_embedded}")
    return images


def describe_image_with_gemini(image: Image.Image, model_name: str) -> str:
    """Use selected Gemini model to describe an image."""
    model = genai.GenerativeModel(model_name)
    prompt = (
        "Describe this image in detail. "
        "If it's a diagram, chart, or technical image, "
        "explain what it shows. "
        "If it contains text, mention the key text content."
    )
    try:
        response = model.generate_content([prompt, image])
        return (
            response.text.strip()
            if response.text
            else "No description available."
        )
    except Exception as e:
        return f"Error generating description: {str(e)}"


def describe_single_image(model_name: str):
    """Describe a single selected image."""
    image_files = list_extracted_images()
    if not image_files:
        return

    while True:
        try:
            choice = input(
                f"\nSelect an image (1-{len(image_files)}) "
                "or 'b' to go back: "
            ).strip()

            if choice.lower() == "b":
                return

            choice = int(choice)

            if 1 <= choice <= len(image_files):
                selected_image_path = image_files[choice - 1]
                print(f"🖼️  Analyzing: {selected_image_path.name}")

                # Load and describe the image
                try:
                    image = Image.open(selected_image_path)
                    print(f"📝 Describing with {model_name}...")
                    description = describe_image_with_gemini(
                        image, model_name
                    )

                    print(f"\n{'=' * 50}")
                    print(f"Image: {selected_image_path.name}")
                    print(f"Model: {model_name}")
                    print(f"{'=' * 50}")
                    print(f"Description: {description}")
                    print(f"{'=' * 50}\n")

                    # Ask if user wants to describe another image
                    another = (
                        input("Describe another image? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if another not in ["y", "yes"]:
                        break

                    # Re-display the image list for next selection
                    print("\n🖼️ Available extracted images:")
                    for img_index, img_file in enumerate(image_files, 1):
                        # Extract info from filename
                        parts = img_file.stem.split("_")
                        if len(parts) >= 4:
                            page_num = parts[1]
                            img_num = parts[3]
                            print(
                                f"  {img_index}. {img_file.parent.name} - "
                                f"Page {page_num}, Image {img_num}"
                            )
                        else:
                            print(f"  {img_index}. {img_file.name}")

                except Exception as e:
                    print(f"❌ Error loading image: {e}")
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'b'.")


def process_pdf_images(pdf_path, model_name: str):
    """Extract images from a PDF (without describing them automatically)."""
    print(f"🚀 Processing PDF: {Path(pdf_path).name}")

    images = extract_images_from_pdf(pdf_path)

    if not images:
        print("⚠️ No embedded images found in this PDF.")
        print("(Note: This PDF might contain vector graphics or text only)")
        return

    print(f"✅ Extraction complete! Found {len(images)} images.")


def save_results(results, pdf_name, model_name):
    """Save results to a text file."""
    result_file = os.path.join(
        RESULTS_PATH,
        f"{Path(pdf_name).stem}_{model_name.replace('-', '_')}_results.txt",
    )

    with open(result_file, "w", encoding="utf-8") as text_file:
        text_file.write(f"Image Analysis Results for: {pdf_name}\n")
        text_file.write(f"Model used: {model_name}\n")
        text_file.write("=" * 50 + "\n\n")

        for result in results:
            text_file.write(
                f"Page {result['page']}, "
                f"Image {result['index']} ({result['type']})\n"
            )
            text_file.write(f"Description: {result['description']}\n")
            if result["image_path"]:
                text_file.write(f"Image saved at: {result['image_path']}\n")
            text_file.write("-" * 30 + "\n")

    print(f"💾 Results saved to: {result_file}")


def main_menu(model_name: str):
    """Display main menu options."""
    while True:
        print(f"\n🎯 Main Menu (Using: {model_name})")
        print("=" * 40)
        print("1. Extract images from PDF")
        print("2. Describe single extracted image")
        print("3. Change model")
        print("4. Quit")

        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            pdf_files = list_available_pdfs()
            if not pdf_files:
                print("Please add PDF files to the 'pdfs' directory.")
                continue

            print(
                f"\nSelect a PDF to extract images from (1-{len(pdf_files)}):"
            )
            try:
                pdf_choice = int(input("> ").strip())
                if 1 <= pdf_choice <= len(pdf_files):
                    selected_pdf = pdf_files[pdf_choice - 1]
                    process_pdf_images(selected_pdf, model_name)
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Invalid input.")

        elif choice == "2":
            describe_single_image(model_name)

        elif choice == "3":
            return "change_model"

        elif choice == "4":
            print("👋 Goodbye!")
            return "quit"

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    print("🖼️  Enhanced PDF Image Extractor & Analyzer")
    print("=" * 50)
    print("📝 This tool extracts EMBEDDED images from PDFs.")

    while True:
        # Model selection
        selected_model = select_model()
        if selected_model is None:
            print("👋 Goodbye!")
            break

        # Main menu loop
        while True:
            result = main_menu(selected_model)
            if result == "change_model":
                break  # Go back to model selection
            elif result == "quit":
                exit()
