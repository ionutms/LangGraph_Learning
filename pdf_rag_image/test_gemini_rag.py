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
MODEL_NAME = "gemini-2.0-flash-lite"

# Directory setup
PDFS_PATH = "./pdf_rag_image/pdfs"
IMAGES_PATH = "./pdf_rag_image/extracted_images"
RESULTS_PATH = "./pdf_rag_image/results"

os.makedirs(PDFS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


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


def describe_image_with_gemini(image: Image.Image) -> str:
    """Use Gemini Flash to describe an image."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "Describe this image in one clear sentence. "
        "If it's a diagram, chart, or technical image, mention that."
    )
    response = model.generate_content([prompt, image])
    return (
        response.text.strip()
        if response.text
        else "No description available."
    )


def process_pdf_images(pdf_path):
    """Process images from a PDF and generate descriptions."""
    print(f"🚀 Processing PDF: {Path(pdf_path).name}")

    images = extract_images_from_pdf(pdf_path)

    if not images:
        print("⚠️ No embedded images found in this PDF.")
        print("(Note: This PDF might contain vector graphics or text only)")

    print(f"🧠 Describing {len(images)} images with Gemini...")
    results = []

    for img_data in images:
        print(
            f"📝 Describing page {img_data['page']}, "
            f"image {img_data['index']} ({img_data['type']})..."
        )
        desc = describe_image_with_gemini(img_data["image"])
        result = {
            "page": img_data["page"],
            "index": img_data["index"],
            "pdf_name": img_data["pdf_name"],
            "type": img_data["type"],
            "description": desc,
            "image_path": img_data["image_path"],
        }
        results.append(result)
        print(f"✅ {desc}")

    # Save results to file
    if results:
        save_results(results, Path(pdf_path).name)
        print(f"✅ Processing complete! Results saved to {RESULTS_PATH}")


def save_results(results, pdf_name):
    """Save results to a text file."""
    result_file = os.path.join(
        RESULTS_PATH, f"{Path(pdf_name).stem}_results.txt"
    )

    with open(result_file, "w", encoding="utf-8") as text_file:
        text_file.write(f"Image Analysis Results for: {pdf_name}\n")
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


if __name__ == "__main__":
    print("🖼️  PDF Image Extractor & Analyzer")
    print("=" * 40)
    print("📝 This tool extracts EMBEDDED images from PDFs only.")
    print("(Vector graphics and text are not converted to images)")

    while True:
        pdf_files = list_available_pdfs()
        if not pdf_files:
            print("Please add PDF files to the 'pdfs' directory.")
            break

        print(
            f"\nSelect a PDF to process (1-{len(pdf_files)}) or 'q' to quit:"
        )
        choice = input("> ").strip()

        if choice.lower() == "q":
            print("👋 Goodbye!")
            break

        choice_num = int(choice)
        if 1 <= choice_num <= len(pdf_files):
            selected_pdf = pdf_files[choice_num - 1]
            process_pdf_images(selected_pdf)
        else:
            print("❌ Invalid selection.")
