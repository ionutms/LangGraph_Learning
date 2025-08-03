"""Vector store initialization and management."""

from config import Config
from file_manager import FileManager
from vector_store import VectorStoreManager


def initialize_vector_store():
    """Initialize and update the vector store with PDF documents."""
    # Create necessary directories
    Config.create_directories()

    # Initialize managers
    file_manager = FileManager()
    vector_manager = VectorStoreManager()

    # Check for changes
    new_or_modified_pdfs, processed_files = (
        file_manager.find_new_or_modified_pdfs()
    )
    deleted_files = file_manager.find_deleted_files(processed_files)

    # Handle deleted files first
    if deleted_files:
        vector_manager.remove_deleted_files(deleted_files)

        # Remove deleted files from processed_files tracking
        for deleted_file in deleted_files:
            processed_files.pop(deleted_file, None)

        # Save updated processed files list
        file_manager.save_processed_files(processed_files)

    # Handle new or modified files
    if new_or_modified_pdfs:
        print(f"Found {len(new_or_modified_pdfs)} new or modified PDF files:")
        for pdf in new_or_modified_pdfs:
            print(f"  - {pdf.name}")

        vector_manager.process_and_add_pdfs(
            new_or_modified_pdfs, processed_files
        )

    elif not deleted_files:
        print("No changes detected in PDF files.")

    # Summary of current state
    current_pdfs = list(Config.PDF_DIRECTORY.glob("*.pdf"))
    print(f"\nCurrent state: {len(current_pdfs)} PDF files in vector store")
    for pdf in current_pdfs:
        print(f"  - {pdf.name}")

    return vector_manager.get_retriever()


# Initialize and get retriever
retriever = initialize_vector_store()
