import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


def extract_table_data_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract table metadata and data from all pages of a PDF.

    Opens the specified PDF and scans each page for detectable tables
    using PyMuPDF's table detection. For each table found, metadata such
    as page number, position (bounding box), dimensions, and cell data
    is collected.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A list of dictionaries, each containing:
            - 'page' (int): 1-based page number.
            - 'table_index' (int): 1-based index of table on the page.
            - 'bbox' (List[float]): Bounding box [x0, y0, x1, y1].
            - 'rows' (int): Number of rows in the table.
            - 'cols' (int): Number of columns in the table.
            - 'data' (List[List[Any]]):
                2D list of cell values (strings, numbers, or None).
    """
    extracted_tables: List[Dict[str, Any]] = []

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


def save_tables_to_csv(
    tables: List[Dict[str, Any]],
    output_directory: str,
) -> None:
    """Save extracted table data to individual CSV files.

    Creates the output directory if it doesn't exist and writes each
    table's data to a separate CSV file named in the format:
    'page_{page}_table_{index}.csv'.

    For each table, None values are converted to empty strings to ensure
    compatibility with CSV format.

    Args:
        tables:
            List of dictionaries containing table metadata and data.
            Each dictionary must include:
                - 'page' (int): 1-based page number
                - 'table_index' (int): 1-based table index on page
                - 'data' (List[List[Any]]): 2D list of table cell values
        output_directory:
            Path to the directory where CSV files will be saved.
            Will be created if it doesn't exist.

    Returns:
        None. However, each table dictionary in the input list will be
        updated with a 'csv_file' key containing the name of the saved file.

    """
    Path(output_directory).mkdir(exist_ok=True)

    for table_info in tables:
        page_number = table_info["page"]
        table_index = table_info["table_index"]
        filename = f"page_{page_number}_table_{table_index}.csv"
        file_path = os.path.join(output_directory, filename)

        table_data = table_info["data"]
        if table_data:
            with open(file_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for row in table_data:
                    cleaned_row = [
                        str(cell) if cell is not None else "" for cell in row
                    ]
                    writer.writerow(cleaned_row)
            table_info["csv_file"] = filename

    print("Table extraction complete!")


def extract_pdf_tables(
    pdf_path: str,
    output_directory: str,
) -> List[Dict[str, Any]]:
    """
    Extract all tables from a PDF and save them as CSV files.

    Args:
        pdf_path: Path to the PDF file.
        output_directory: Directory to save extracted CSVs.

    Returns:
        List of extracted table metadata and data.
    """
    tables = extract_table_data_from_pdf(pdf_path)
    save_tables_to_csv(tables, output_directory)
    return tables


def list_tables_in_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """List metadata of all tables in the PDF without extracting content.

    Opens the specified PDF and scans each page for detectable tables
    using PyMuPDF's table detection. For each table found, metadata such
    as page number, position (bounding box), and dimensions is collected.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dictionaries, each containing:
            - 'page' (int): 1-based page number.
            - 'table_index' (int): 0-based index of table on the page.
            - 'rows' (int): Number of rows in the table.
            - 'cols' (int): Number of columns in the table.
            - 'bbox' (List[float]): Bounding box [x0, y0, x1, y1].
    """
    tables_info: List[Dict[str, Any]] = []

    with fitz.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document[page_index]

            tables = page.find_tables()
            for table_idx, table in enumerate(tables):
                table_data = table.extract()
                num_rows = len(table_data)
                num_cols = len(table_data[0]) if table_data else 0

                tables_info.append({
                    "page": page_index + 1,
                    "table_index": table_idx,
                    "rows": num_rows,
                    "cols": num_cols,
                    "bbox": [float(coord) for coord in table.bbox],
                })

    return tables_info


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))

    pdfs_directory = os.path.join(script_directory, "pdfs")
    search_directory = (
        pdfs_directory
        if os.path.exists(pdfs_directory) and os.path.isdir(pdfs_directory)
        else script_directory
    )

    print(f"Searching for PDFs in: {search_directory}")

    pdf_filenames = [
        filename
        for filename in os.listdir(search_directory)
        if filename.lower().endswith(".pdf")
    ]
    pdf_file_paths = [
        os.path.join(search_directory, filename) for filename in pdf_filenames
    ]

    if not pdf_file_paths:
        print(f"No PDF files found in {search_directory}")
        print("You can either:")
        print("1. Place PDF files in the script directory")
        print("2. Create a 'pdfs' subdirectory and place files there")
        exit(1)

    # Main output directory
    main_output_dir = os.path.join(script_directory, "extracted_tables")
    os.makedirs(main_output_dir, exist_ok=True)

    # Process all PDFs
    for pdf_path in pdf_file_paths:
        pdf_name = Path(pdf_path).stem  # e.g., "document" from "document.pdf"
        pdf_output_dir = os.path.join(main_output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        print(f"Saving tables to: {pdf_output_dir}")

        # Extract and save all tables
        extracted_tables = extract_pdf_tables(pdf_path, pdf_output_dir)

        # Show overview
        print("\nTable overview:")
        table_list = list_tables_in_pdf(pdf_path)
        if not table_list:
            print("  No tables found in the PDF.")
        else:
            for info in table_list:
                print(
                    f"  Page {info['page']}, "
                    f"Table {info['table_index'] + 1}: "
                    f"{info['rows']}×{info['cols']} cells"
                )
