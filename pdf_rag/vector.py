import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
PDF_DIRECTORY = "./pdf_rag/pdfs"
db_location = "./pdf_rag/pdf_rag_agent/chroma_langchain_db"
metadata_file = "./pdf_rag/pdf_rag_agent/processed_files.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Create directories if they don't exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(db_location), exist_ok=True)


def load_processed_files():
    """Load the list of previously processed files"""
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files):
    """Save the list of processed files"""
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(processed_files, f, indent=2)


def get_file_info(file_path):
    """Get file modification time and size for change detection"""
    stat = os.stat(file_path)
    return {"modified_time": stat.st_mtime, "size": stat.st_size}


def find_new_or_modified_pdfs():
    """Find PDFs that are new or have been modified"""
    processed_files = load_processed_files()
    pdf_files = list(Path(PDF_DIRECTORY).glob("*.pdf"))

    new_or_modified = []

    for pdf_path in pdf_files:
        file_key = str(pdf_path)
        current_info = get_file_info(pdf_path)

        # Check if file is new or modified
        if (
            file_key not in processed_files
            or processed_files[file_key]["modified_time"]
            != current_info["modified_time"]
            or processed_files[file_key]["size"] != current_info["size"]
        ):
            new_or_modified.append(pdf_path)

    return new_or_modified, processed_files


def find_deleted_files(processed_files):
    """Find files that were processed before but no longer exist"""
    current_pdf_files = set(
        str(pdf) for pdf in Path(PDF_DIRECTORY).glob("*.pdf")
    )
    processed_file_paths = set(processed_files.keys())

    deleted_files = processed_file_paths - current_pdf_files
    return deleted_files


def remove_deleted_files_from_vectorstore(vector_store, deleted_files):
    """Remove chunks from deleted files from the vector store"""
    if not deleted_files:
        return

    print(f"Removing {len(deleted_files)} deleted files from vector store:")
    for deleted_file in deleted_files:
        filename = Path(deleted_file).name
        print(f"  - {filename}")

    # Get all documents from vector store
    all_docs = vector_store.get()

    if not all_docs or not all_docs["metadatas"]:
        return

    # Find IDs of documents from deleted files
    ids_to_delete = []
    deleted_filenames = {Path(f).name for f in deleted_files}

    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and "source_file" in metadata:
            if metadata["source_file"] in deleted_filenames:
                ids_to_delete.append(all_docs["ids"][i])

    # Delete the documents
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
        print(f"Removed {len(ids_to_delete)} chunks from deleted files")


# Initialize embeddings (needed for vector store operations)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
)

# Initialize vector store
vector_store = Chroma(
    collection_name="pdf_documents",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Check for new, modified, and deleted PDFs
new_or_modified_pdfs, processed_files = find_new_or_modified_pdfs()
deleted_files = find_deleted_files(processed_files)

# Handle deleted files first
if deleted_files:
    remove_deleted_files_from_vectorstore(vector_store, deleted_files)

    # Remove deleted files from processed_files tracking
    for deleted_file in deleted_files:
        if deleted_file in processed_files:
            del processed_files[deleted_file]

    # Save updated processed files list
    save_processed_files(processed_files)

# Handle new or modified files
if new_or_modified_pdfs:
    print(f"Found {len(new_or_modified_pdfs)} new or modified PDF files:")
    for pdf in new_or_modified_pdfs:
        print(f"  - {pdf.name}")

    documents = []

    # For modified files, we need to remove old chunks first
    for pdf_path in new_or_modified_pdfs:
        file_key = str(pdf_path)

        # If this is a modified file (not new), remove old chunks
        if file_key in processed_files:
            print(f"Removing old chunks for modified file: {pdf_path.name}")

            # Get all documents and find ones from this specific file
            all_docs = vector_store.get()
            if all_docs and all_docs["metadatas"]:
                ids_to_delete = []
                for i, metadata in enumerate(all_docs["metadatas"]):
                    if (
                        metadata
                        and "source_file" in metadata
                        and metadata["source_file"] == pdf_path.name
                    ):
                        ids_to_delete.append(all_docs["ids"][i])

                if ids_to_delete:
                    vector_store.delete(ids=ids_to_delete)
                    print(f"Removed {len(ids_to_delete)} old chunks")

        print(f"Processing: {pdf_path.name}")

        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        # Add source filename to metadata
        for doc in pdf_documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["file_path"] = str(pdf_path)

        documents.extend(pdf_documents)

        # Update processed files tracking
        processed_files[str(pdf_path)] = get_file_info(pdf_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    print("Splitting documents into chunks...")
    split_documents = text_splitter.split_documents(documents)
    print(f"Created {len(split_documents)} document chunks")

    print("Adding new/updated documents to vector store...")
    vector_store.add_documents(documents=split_documents)

    # Save updated processed files list
    save_processed_files(processed_files)
    print("Documents updated successfully!")

elif not deleted_files:
    print("No changes detected in PDF files.")

# Summary of current state
current_pdfs = list(Path(PDF_DIRECTORY).glob("*.pdf"))
print(f"\nCurrent state: {len(current_pdfs)} PDF files in vector store")
for pdf in current_pdfs:
    print(f"  - {pdf.name}")

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 20})
