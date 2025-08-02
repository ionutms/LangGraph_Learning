import os
from pathlib import Path

import fitz  # PyMuPDF

# Directory setup
PDFS_PATH = "./pdf_rag_image/pdfs"
IMAGES_PATH = "./pdf_rag_image/extracted_images"

os.makedirs(PDFS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)


def list_available_pdfs():
    """List all PDF files in the pdfs directory."""
    pdf_files = list(Path(PDFS_PATH).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in {PDFS_PATH}")
        print("Please add PDF files to the 'pdfs' directory.")
        return []

    print("📚 Available PDF files:")
    for pdf_index, pdf_file in enumerate(pdf_files, 1):
        print(f"  {pdf_index}. {pdf_file.name}")
    return pdf_files


def extract_images_from_pdf(pdf_path, output_folder=None):
    """Extract all images from a PDF file."""
    pdf_name = Path(pdf_path).name
    print(f"🔍 Extracting images from: {pdf_name}")

    # Create output folder specific to this PDF
    if output_folder is None:
        pdf_folder = os.path.join(IMAGES_PATH, Path(pdf_path).stem)
    else:
        pdf_folder = output_folder

    os.makedirs(pdf_folder, exist_ok=True)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    image_count = 0

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Get list of images on this page
        image_list = page.get_images()

        if image_list:
            print(f"📄 Page {page_num + 1}: Found {len(image_list)} images")

        # Extract each image
        for img_index, img in enumerate(image_list):
            # Get image data
            xref = img[0]  # xref number
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # file extension

            # Save the image
            image_filename = (
                f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            )
            image_path = os.path.join(pdf_folder, image_filename)

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            image_count += 1
            print(f"💾 Saved: {image_filename} (format: {image_ext})")

    print(f"📊 Total images extracted: {image_count}")
    if image_count > 0:
        print(f"📁 Images saved in: {pdf_folder}")

    pdf_document.close()
    return image_count


if __name__ == "__main__":
    print("🖼️ PDF Image Extractor")
    print("=" * 30)
    print("📝 This tool extracts all embedded images from PDF files.")

    while True:
        print()
        pdf_files = list_available_pdfs()
        if not pdf_files:
            break

        print(
            f"\nSelect a PDF to process (1-{len(pdf_files)}) or 'q' to quit:"
        )
        choice = input("> ").strip()

        if choice.lower() == "q":
            print("👋 Goodbye!")
            break

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(pdf_files):
                selected_pdf = pdf_files[choice_num - 1]
                print(f"\n🚀 Processing: {selected_pdf.name}")

                image_count = extract_images_from_pdf(selected_pdf)

                if image_count == 0:
                    print("⚠️ No images found in this PDF.")
                    print(
                        "This PDF might contain only text or vector graphics"
                    )
                else:
                    print(f"✅ Successfully extracted {image_count} images!")

            else:
                print("❌ Invalid selection. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit.")
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
