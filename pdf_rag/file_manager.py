"""File management utilities for PDF tracking."""

import json
import os
from pathlib import Path
from typing import Dict, List, Set

from config import Config, FileInfo


class FileManager:
    """Manages PDF file tracking and metadata."""

    def __init__(self) -> None:
        self.metadata_file = Config.METADATA_FILE
        self.pdf_directory = Config.PDF_DIRECTORY

    def load_processed_files(self) -> Dict[str, FileInfo]:
        """Load the list of previously processed files."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def save_processed_files(
        self, processed_files: Dict[str, FileInfo]
    ) -> None:
        """Save the list of processed files."""
        with open(self.metadata_file, "w") as f:
            json.dump(processed_files, f, indent=2)

    def get_file_info(self, file_path: Path) -> FileInfo:
        """Get file modification time and size for change detection."""
        stat = os.stat(file_path)
        return FileInfo(modified_time=stat.st_mtime, size=stat.st_size)

    def find_new_or_modified_pdfs(
        self,
    ) -> tuple[List[Path], Dict[str, FileInfo]]:
        """Find PDFs that are new or have been modified."""
        processed_files = self.load_processed_files()
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        new_or_modified = []

        for pdf_path in pdf_files:
            file_key = str(pdf_path)
            current_info = self.get_file_info(pdf_path)

            # Check if file is new or modified
            if self._is_file_changed(file_key, current_info, processed_files):
                new_or_modified.append(pdf_path)

        return new_or_modified, processed_files

    def find_deleted_files(
        self, processed_files: Dict[str, FileInfo]
    ) -> Set[str]:
        """Find files that were processed before but no longer exist."""
        current_pdf_files = {
            str(pdf) for pdf in self.pdf_directory.glob("*.pdf")
        }
        processed_file_paths = set(processed_files.keys())
        return processed_file_paths - current_pdf_files

    def _is_file_changed(
        self,
        file_key: str,
        current_info: FileInfo,
        processed_files: Dict[str, FileInfo],
    ) -> bool:
        """Check if a file is new or has been modified."""
        if file_key not in processed_files:
            return True

        old_info = processed_files[file_key]
        return (
            old_info["modified_time"] != current_info["modified_time"]
            or old_info["size"] != current_info["size"]
        )
