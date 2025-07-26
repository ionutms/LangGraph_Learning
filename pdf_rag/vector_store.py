"""Vector store operations for PDF documents."""

from pathlib import Path
from typing import Dict, List, Set

from config import Config, FileInfo
from file_manager import FileManager
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """Manages vector store operations."""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=Config.EMBEDDING_MODEL,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
        )

        self.vector_store = Chroma(
            collection_name="pdf_documents",
            persist_directory=str(Config.DB_LOCATION),
            embedding_function=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self.file_manager = FileManager()

    def remove_deleted_files(self, deleted_files: Set[str]) -> None:
        """Remove chunks from deleted files from the vector store."""
        if not deleted_files:
            return

        print(
            f"Removing {len(deleted_files)} deleted files from vector store:"
        )
        for deleted_file in deleted_files:
            filename = Path(deleted_file).name
            print(f"  - {filename}")

        # Get all documents from vector store
        all_docs = self.vector_store.get()

        if not all_docs or not all_docs["metadatas"]:
            return

        # Find IDs of documents from deleted files
        ids_to_delete = self._find_ids_by_filenames(
            all_docs, {Path(f).name for f in deleted_files}
        )

        # Delete the documents
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} chunks from deleted files")

    def remove_old_chunks(self, filename: str) -> None:
        """Remove old chunks for a modified file."""
        print(f"Removing old chunks for modified file: {filename}")

        all_docs = self.vector_store.get()
        if not all_docs or not all_docs["metadatas"]:
            return

        ids_to_delete = self._find_ids_by_filenames(all_docs, {filename})

        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} old chunks")

    def process_and_add_pdfs(
        self, pdf_paths: List[Path], processed_files: Dict[str, FileInfo]
    ) -> None:
        """Process and add PDF documents to vector store."""
        documents = []

        for pdf_path in pdf_paths:
            file_key = str(pdf_path)

            # If this is a modified file, remove old chunks first
            if file_key in processed_files:
                self.remove_old_chunks(pdf_path.name)

            print(f"Processing: {pdf_path.name}")

            # Load and process PDF
            pdf_documents = self._load_pdf(pdf_path)
            documents.extend(pdf_documents)

            # Update processed files tracking
            processed_files[file_key] = self.file_manager.get_file_info(
                pdf_path
            )

        # Split and add documents
        self._split_and_add_documents(documents)

        # Save updated processed files list
        self.file_manager.save_processed_files(processed_files)
        print("Documents updated successfully!")

    def get_retriever(self):
        """Get retriever for the vector store."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )

    def _find_ids_by_filenames(
        self, all_docs: dict, filenames: Set[str]
    ) -> List[str]:
        """Find document IDs by filenames."""
        ids_to_delete = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if (
                metadata
                and "source_file" in metadata
                and metadata["source_file"] in filenames
            ):
                ids_to_delete.append(all_docs["ids"][i])

        return ids_to_delete

    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a PDF file and add metadata."""
        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        # Add source filename to metadata
        for doc in pdf_documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["file_path"] = str(pdf_path)

        return pdf_documents

    def _split_and_add_documents(self, documents: List[Document]) -> None:
        """Split documents into chunks and add to vector store."""
        print("Splitting documents into chunks...")
        split_documents = self.text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} document chunks")

        print("Adding new/updated documents to vector store...")
        self.vector_store.add_documents(documents=split_documents)
