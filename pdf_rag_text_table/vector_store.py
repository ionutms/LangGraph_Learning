"""Vector store operations for PDF documents."""

import csv
from pathlib import Path
from typing import Dict, List, Set

import fitz  # PyMuPDF
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

            # Load and process PDF text
            pdf_documents = self._load_pdf(pdf_path)
            documents.extend(pdf_documents)

            # Extract and process tables
            table_documents = self._extract_and_process_tables(pdf_path)
            documents.extend(table_documents)

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

    def _extract_table_data_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract table metadata and data from all pages of a PDF."""
        extracted_tables = []

        with fitz.open(pdf_path) as document:
            for page_number in range(document.page_count):
                page = document[page_number]

                tables = page.find_tables()
                for table_index, table in enumerate(tables):
                    table_data = table.extract()
                    num_rows = len(table_data)
                    num_cols = len(table_data[0]) if table_data else 0

                    table_info = {
                        "page": page_number + 1,
                        "table_index": table_index + 1,
                        "bbox": [float(coord) for coord in table.bbox],
                        "rows": num_rows,
                        "cols": num_cols,
                        "data": table_data,
                    }
                    extracted_tables.append(table_info)

        return extracted_tables

    def _format_table_as_text(self, table_data: List[List]) -> str:
        """Format table data as structured text for embedding."""
        if not table_data:
            return ""

        # Convert None values to empty strings
        cleaned_data = []
        for row in table_data:
            cleaned_row = [
                str(cell) if cell is not None else "" for cell in row
            ]
            cleaned_data.append(cleaned_row)

        # Create text representation
        text_lines = []
        for i, row in enumerate(cleaned_data):
            if i == 0:  # Header row
                text_lines.append("Table Headers: " + " | ".join(row))
                text_lines.append("-" * 40)
            else:
                text_lines.append("Row " + str(i) + ": " + " | ".join(row))

        return "\n".join(text_lines)

    def _save_table_to_csv(self, table_info: Dict, output_dir: Path) -> str:
        """Save individual table to CSV file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        page_num = table_info["page"]
        table_idx = table_info["table_index"]
        filename = f"page_{page_num}_table_{table_idx}.csv"
        file_path = output_dir / filename

        table_data = table_info["data"]
        if table_data:
            with open(file_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for row in table_data:
                    cleaned_row = [
                        str(cell) if cell is not None else "" for cell in row
                    ]
                    writer.writerow(cleaned_row)

        return filename

    def _extract_tables_as_documents(self, pdf_path: Path) -> List[Document]:
        """Extract tables from PDF and return as Document objects."""
        documents = []
        tables = self._extract_table_data_from_pdf(str(pdf_path))

        for table_info in tables:
            if table_info["data"]:
                # Format table as text
                table_text = self._format_table_as_text(table_info["data"])

                # Create document with table content
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source_file": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page": table_info["page"],
                        "table_index": table_info["table_index"],
                        "content_type": "table",
                        "rows": table_info["rows"],
                        "cols": table_info["cols"],
                        "bbox": str(table_info["bbox"]),  # Convert to string
                    },
                )
                documents.append(doc)

        return documents

    def _extract_and_process_tables(self, pdf_path: Path) -> List[Document]:
        """Extract tables from PDF and create table directory."""
        # Create table output directory for this PDF
        pdf_table_dir = Config.TABLES_DIRECTORY / pdf_path.stem
        pdf_table_dir.mkdir(parents=True, exist_ok=True)

        # Extract tables as documents
        table_documents = self._extract_tables_as_documents(pdf_path)

        # Also save tables as CSV files for reference
        tables = self._extract_table_data_from_pdf(str(pdf_path))
        for table_info in tables:
            self._save_table_to_csv(table_info, pdf_table_dir)

        if table_documents:
            print(f"  Found {len(table_documents)} tables")

        return table_documents

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

        # Add source filename and content type to metadata
        for doc in pdf_documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["file_path"] = str(pdf_path)
            doc.metadata["content_type"] = "text"

        return pdf_documents

    def _filter_metadata(self, metadata: dict) -> dict:
        """Filter metadata to only include simple types for Chroma."""
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, list):
                # Convert lists to strings
                filtered[key] = str(value)
            else:
                # Convert other types to strings
                filtered[key] = str(value)
        return filtered

    def _split_and_add_documents(self, documents: List[Document]) -> None:
        """Split documents into chunks and add to vector store."""
        # Separate text and table documents
        text_docs = [
            d for d in documents if d.metadata.get("content_type") == "text"
        ]
        table_docs = [
            d for d in documents if d.metadata.get("content_type") == "table"
        ]

        all_chunks = []

        # Split text documents
        if text_docs:
            print("Splitting text documents into chunks...")
            text_chunks = self.text_splitter.split_documents(text_docs)
            # Filter metadata from text chunks
            for doc in text_chunks:
                filtered_doc = Document(
                    page_content=doc.page_content,
                    metadata=self._filter_metadata(doc.metadata),
                )
                all_chunks.append(filtered_doc)

        # Add table documents as-is (don't split them)
        if table_docs:
            print(f"Adding {len(table_docs)} table documents...")
            # Filter metadata from table docs
            for doc in table_docs:
                filtered_doc = Document(
                    page_content=doc.page_content,
                    metadata=self._filter_metadata(doc.metadata),
                )
                all_chunks.append(filtered_doc)

        print(f"Created {len(all_chunks)} document chunks total")

        print("Adding new/updated documents to vector store...")
        self.vector_store.add_documents(documents=all_chunks)
