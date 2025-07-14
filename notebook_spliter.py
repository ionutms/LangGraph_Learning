import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph


class NotebookSplitterState(TypedDict):
    input_notebook_path: str
    output_directory: str
    notebook_content: Dict[str, Any]
    sections: List[Dict[str, Any]]
    created_notebooks: List[str]
    error_message: str
    split_level: int


def load_notebook(state: NotebookSplitterState) -> NotebookSplitterState:
    """Load the input Jupyter notebook.

    Args:
        state: The current state containing the notebook path and other
            workflow data.

    Returns:
        Updated state with loaded notebook content or error message.
    """
    try:
        with open(
            state["input_notebook_path"], "r", encoding="utf-8"
        ) as file:
            notebook_content = json.load(file)

        state["notebook_content"] = notebook_content
        print(f"✓ Loaded: {state['input_notebook_path']}")
        return state
    except Exception as error:
        state["error_message"] = f"Load error: {str(error)}"
        return state


def extract_sections(state: NotebookSplitterState) -> NotebookSplitterState:
    """Extract sections based on markdown headings at specified level.

    Args:
        state: The current state containing notebook content and split level.

    Returns:
        Updated state with extracted sections or error message.
    """
    try:
        notebook = state["notebook_content"]
        cells = notebook.get("cells", [])
        split_level = state["split_level"]
        sections = []
        current_section = {"title": "Introduction", "cells": [], "level": 0}

        for cell in cells:
            if cell.get("cell_type") == "markdown":
                # Check if this is a heading cell
                source = "".join(cell.get("source", []))
                heading_match = re.match(r"^(#{1,6})\s+(.+)", source.strip())

                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    # Only split on headings at or above the specified level
                    if level <= split_level:
                        # Save current section if it has content
                        if current_section["cells"]:
                            sections.append(current_section)

                        # Start new section
                        current_section = {
                            "title": title,
                            "cells": [cell],
                            "level": level,
                        }
                    else:
                        # Lower level heading, add to current section
                        current_section["cells"].append(cell)
                else:
                    current_section["cells"].append(cell)
            else:
                current_section["cells"].append(cell)

        # Add the last section
        if current_section["cells"]:
            sections.append(current_section)

        state["sections"] = sections
        print(f"✓ Found {len(sections)} sections (H{split_level})")
        return state

    except Exception as error:
        state["error_message"] = f"Extract error: {str(error)}"
        return state


def create_output_notebooks(
    state: NotebookSplitterState,
) -> NotebookSplitterState:
    """Create individual notebooks for each section.

    Args:
        state: The current state containing sections and output directory.

    Returns:
        Updated state with created notebook paths or error message.
    """
    try:
        original_notebook = state["notebook_content"]
        output_dir = Path(state["output_directory"])
        output_dir.mkdir(exist_ok=True)

        created_notebooks = []

        for section_index, section in enumerate(state["sections"]):
            # Create new notebook structure
            new_notebook = {
                "cells": section["cells"],
                "metadata": original_notebook.get("metadata", {}),
                "nbformat": original_notebook.get("nbformat", 4),
                "nbformat_minor": original_notebook.get("nbformat_minor", 2),
            }

            # Generate filename
            safe_title = re.sub(r"[^\w\s-]", "", section["title"])
            safe_title = re.sub(r"[-\s]+", "_", safe_title)
            filename = f"{safe_title}.ipynb"
            output_path = output_dir / filename

            # Write notebook
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(new_notebook, file, indent=2)

            created_notebooks.append(str(output_path))
            print(f"✓ Created: {filename}")

        state["created_notebooks"] = created_notebooks
        return state

    except Exception as error:
        state["error_message"] = f"Create error: {str(error)}"
        return state


def check_errors(state: NotebookSplitterState) -> str:
    """Check if there are any errors in the state.

    Args:
        state: The current state to check for errors.

    Returns:
        "error" if there are errors, "success" otherwise.
    """
    return "error" if state.get("error_message") else "success"


def handle_error(state: NotebookSplitterState) -> NotebookSplitterState:
    """Handle errors by printing error message.

    Args:
        state: The current state containing error information.

    Returns:
        The unchanged state after handling the error.
    """
    print(f"❌ {state['error_message']}")
    return state


def finalize_success(state: NotebookSplitterState) -> NotebookSplitterState:
    """Finalize successful processing by printing success message.

    Args:
        state: The current state containing successful processing results.

    Returns:
        The unchanged state after printing success information.
    """
    print(f"\n✅ Created {len(state['created_notebooks'])} notebooks")
    print(f"📁 Output: {state['output_directory']}")
    return state


class NotebookSplitter:
    """Main class for the notebook splitter application.

    This class provides functionality to split Jupyter notebooks into smaller
    notebooks based on markdown heading levels using a LangGraph workflow.
    """

    def __init__(self):
        """Initialize the NotebookSplitter with a compiled workflow."""
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow.

        Returns:
            Compiled StateGraph workflow for notebook splitting.
        """
        workflow = StateGraph(NotebookSplitterState)

        # Add nodes
        workflow.add_node("load_notebook", load_notebook)
        workflow.add_node("extract_sections", extract_sections)
        workflow.add_node("create_notebooks", create_output_notebooks)
        workflow.add_node("handle_error", handle_error)
        workflow.add_node("finalize_success", finalize_success)

        # Add edges
        workflow.add_edge("load_notebook", "extract_sections")
        workflow.add_edge("extract_sections", "create_notebooks")

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "load_notebook",
            check_errors,
            {"error": "handle_error", "success": "extract_sections"},
        )

        workflow.add_conditional_edges(
            "extract_sections",
            check_errors,
            {"error": "handle_error", "success": "create_notebooks"},
        )

        workflow.add_conditional_edges(
            "create_notebooks",
            check_errors,
            {"error": "handle_error", "success": "finalize_success"},
        )

        # Set entry point and end nodes
        workflow.set_entry_point("load_notebook")
        workflow.add_edge("handle_error", END)
        workflow.add_edge("finalize_success", END)

        return workflow.compile()

    def split_notebook(
        self,
        input_path: str,
        output_dir: str = "split_notebooks",
        split_level: int = 1,
    ) -> Dict[str, Any]:
        """Split a notebook into multiple notebooks based on headings.

        Args:
            input_path: Path to the input Jupyter notebook file.
            output_dir: Directory to save the split notebooks. Defaults to
                "split_notebooks".
            split_level: Heading level to split on (1-6). Defaults to 1.

        Returns:
            Dictionary containing the final workflow state with results or
            error information.
        """
        initial_state = NotebookSplitterState(
            input_notebook_path=input_path,
            output_directory=output_dir,
            notebook_content={},
            sections=[],
            created_notebooks=[],
            error_message="",
            split_level=split_level,
        )

        result = self.workflow.invoke(initial_state)
        return result

    def view_graph(self):
        """Display the workflow graph structure in ASCII format."""
        print("\n📊 Workflow Graph:")
        print("=" * 40)
        print(self.workflow.get_graph().draw_ascii())
        print("=" * 40)


def discover_notebooks(directory: str = ".") -> List[str]:
    """Discover all Jupyter notebooks in the specified directory and subdirs.

    Args:
        directory: Directory to search for notebooks. Defaults to current
            directory.

    Returns:
        Sorted list of notebook file paths found in the directory tree.
    """
    notebook_files = []
    for root, directories, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb") and not file.startswith("."):
                notebook_files.append(os.path.join(root, file))
    return sorted(notebook_files)


def select_notebook_interactive() -> str:
    """Interactive notebook selection from current directory.

    Automatically discovers notebooks in current directory and presents them
    for user selection.

    Returns:
        Path to the selected notebook, or empty string if none selected.
    """
    print("📚 Finding notebooks...")

    # Discover in current directory
    notebooks = discover_notebooks(".")
    if not notebooks:
        print("❌ No notebooks found.")
        return ""

    return display_and_select_notebooks(notebooks)


def display_and_select_notebooks(notebooks: List[str]) -> str:
    """Display discovered notebooks and let user select one.

    Args:
        notebooks: List of notebook file paths to display.

    Returns:
        Path to the selected notebook, or empty string if invalid selection.
    """
    print(f"\n📖 Found {len(notebooks)} notebook(s):")
    print("-" * 40)

    for index, notebook in enumerate(notebooks, 1):
        # Get file size
        try:
            size = os.path.getsize(notebook)
            size_str = (
                f"{size / 1024:.1f} KB"
                if size < 1024 * 1024
                else f"{size / (1024 * 1024):.1f} MB"
            )
        except OSError:
            size_str = "Unknown"

        # Get modification time
        try:
            mtime = os.path.getmtime(notebook)
            from datetime import datetime

            mod_time = datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except OSError:
            mod_time = "Unknown"

        print(f"{index:2d}. {notebook}")
        print(f"     {size_str} | {mod_time}")
        print()

    print("-" * 40)

    try:
        selection = int(input(f"Select (1-{len(notebooks)}): ").strip())
        if 1 <= selection <= len(notebooks):
            return notebooks[selection - 1]
        else:
            print("❌ Invalid selection.")
            return ""
    except ValueError:
        print("❌ Enter a number.")
        return ""


def preview_notebook_structure(
    notebook_path: str, split_level: int = 1
) -> None:
    """Preview the notebook structure showing headings and split points.

    Args:
        notebook_path: Path to the notebook file to preview.
        split_level: Heading level to split on for preview. Defaults to 1.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as file:
            notebook = json.load(file)

        print(f"\n📋 Structure: {os.path.basename(notebook_path)}")
        print(f"🔀 Split Level: H{split_level}")
        print("=" * 50)

        headings = []
        split_points = []

        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                source = "".join(cell.get("source", []))
                heading_match = re.match(r"^(#{1,6})\s+(.+)", source.strip())
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    headings.append((level, title))

                    # Mark split points
                    is_split_point = level <= split_level
                    if is_split_point:
                        split_points.append((level, title))

                    # Display with visual indicators
                    indent = "  " * (level - 1)
                    split_indicator = "📄 " if is_split_point else "   "
                    print(f"{split_indicator}{indent}{'#' * level} {title}")

        if not headings:
            print("❌ No headings found.")
            return

        print("\n" + "=" * 50)
        print("📊 Summary:")
        print(f"   • Headings: {len(headings)}")
        print(f"   • Split points: {len(split_points)}")
        print(f"   • Notebooks: ~{len(split_points)}")

        # Show heading level distribution
        level_counts = {}
        for level, title in headings:
            level_counts[level] = level_counts.get(level, 0) + 1

        print("   • Levels:")
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            split_marker = " (SPLIT)" if level <= split_level else ""
            print(f"     H{level}: {count}{split_marker}")

        print("=" * 50)

    except Exception as error:
        print(f"❌ Preview error: {error}")


def select_split_level() -> int:
    """Interactive selection of heading level for splitting.

    Returns:
        Selected heading level (1-6) for splitting notebooks.
    """
    print("\n🎯 Split Level:")
    print("  1. H1 only - Fewer sections")
    print("  2. H1-H2 - More sections")
    print("  3. H1-H3")
    print("  4. H1-H4")
    print("  5. H1-H5")
    print("  6. All levels - Most sections")

    while True:
        try:
            choice = input("\nSelect (1-6, default 1): ").strip()
            if not choice:
                return 1

            level = int(choice)
            if 1 <= level <= 6:
                return level
            else:
                print("❌ Enter 1-6.")
        except ValueError:
            print("❌ Enter a number.")


def explain_split_levels():
    """Explain what different split levels mean."""
    print("\n💡 Split Levels:")
    print("=" * 30)
    print("Level 1: Split on # only")
    print("Level 2: Split on # and ##")
    print("Level 3: Split on # to ###")
    print("Higher = More notebooks")
    print("Lower = Fewer notebooks")
    print("=" * 30)


if __name__ == "__main__":
    """Interactive usage of the notebook splitter.

    Main entry point that provides a command-line interface for splitting
    Jupyter notebooks with interactive selection and configuration options.
    """
    print("🚀 Notebook Splitter")
    print("=" * 30)

    splitter = NotebookSplitter()

    # Display the workflow graph at startup
    splitter.view_graph()

    print("\n" + "=" * 30)
    print("Starting process...")
    print("=" * 30)

    # Interactive notebook selection
    input_notebook = select_notebook_interactive()

    if not input_notebook:
        print("❌ No notebook selected.")
        exit()

    # Validate input file
    if not os.path.exists(input_notebook):
        print(f"❌ File not found: {input_notebook}")
        exit()

    if not input_notebook.endswith(".ipynb"):
        print("❌ Not a .ipynb file")
        exit()

    # Select split level
    print("\n" + "=" * 30)
    explain_split_levels()
    split_level = select_split_level()

    # Preview notebook structure with split level
    preview_notebook_structure(input_notebook, split_level)

    # Ask for confirmation
    proceed = (
        input(f"\nProceed with H{split_level} split? (y/N): ").strip().lower()
    )
    if proceed not in ["y", "yes"]:
        print("❌ Cancelled.")
        exit()

    # Get output directory from user (optional)
    output_directory = input(
        "\nOutput dir (default: split_notebooks): "
    ).strip()
    if not output_directory:
        output_directory = "split_notebooks"

    print("\n🚀 Starting split...")
    print(f"📖 Input: {input_notebook}")
    print(f"🔀 Level: H{split_level}")
    print(f"📁 Output: {output_directory}")
    print("-" * 30)

    # Split the notebook
    result = splitter.split_notebook(
        input_notebook, output_directory, split_level
    )

    if result.get("error_message"):
        print(f"❌ Failed: {result['error_message']}")
    else:
        print(f"\n✅ Success! {len(result['created_notebooks'])} notebooks:")
        for notebook in result["created_notebooks"]:
            print(f"  - {notebook}")
        print(f"\n📂 Saved in: {output_directory}")
