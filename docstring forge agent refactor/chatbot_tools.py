from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def model_selection_tool(
    available_models: List[str], current_model: str = ""
) -> Dict[str, Any]:
    """Tool for selecting an LLM model from available options.

    Args:
        available_models: List of available model identifiers.
        current_model: Currently selected model (if any).

    Returns:
        Dict: Contains 'selected_model' and 'error'.
    """
    try:
        if not available_models:
            return {
                "selected_model": "",
                "error": "No models available for selection",
            }

        print("\nAvailable LLM Models:")
        for model_index, model in enumerate(available_models, 1):
            current_indicator = " (current)" if model == current_model else ""
            print(f"{model_index}. {model}{current_indicator}")

        while True:
            choice = input(
                f"\nSelect a model (1-{len(available_models)}): "
            ).strip()

            try:
                selected_index = int(choice) - 1
                if 0 <= selected_index < len(available_models):
                    selected_model = available_models[selected_index]
                    print(f"\n🤖 Selected: {selected_model}")
                    print("=" * 60)
                    return {
                        "selected_model": selected_model,
                        "error": "",
                    }
                print(
                    f"Select a number between 1 and {len(available_models)}."
                )
            except ValueError:
                print("Invalid input. Please enter a number.")

    except KeyboardInterrupt:
        return {
            "selected_model": "",
            "error": "Model selection cancelled by user",
        }
    except Exception as e:
        return {
            "selected_model": "",
            "error": f"Error in model selection: {str(e)}",
        }


@tool
def continue_prompt_tool() -> Dict[str, Any]:
    """Tool for asking user whether to continue chatting.

    Returns:
        Dict: Contains 'continue' (bool), 'user_input', and 'error'.
    """
    try:
        choice = input("\n🔄 Continue chatting? (y/n): ").lower().strip()

        continue_chatting = choice in ["y", "yes"]

        return {
            "continue": continue_chatting,
            "user_input": choice,
            "error": "",
        }

    except KeyboardInterrupt:
        return {
            "continue": False,
            "user_input": "",
            "error": "Continue prompt cancelled by user",
        }
    except Exception as e:
        return {
            "continue": False,
            "user_input": "",
            "error": f"Error asking to continue: {str(e)}",
        }


@tool
def find_python_files_tool(directory: str) -> Dict[str, Any]:
    """Find all Python files in directory and subdirectories.

    Args:
        directory: Path string for the directory to search.

    Returns:
        Dict: Contains 'python_files' (list of file paths) and 'error'.
    """
    try:
        directory_path = Path(directory)
        python_files = []
        skip_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            "env",
            ".env",
            "node_modules",
            ".tox",
            "build",
            "dist",
        }
        for py_file in directory_path.rglob("*.py"):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            if any(part.startswith(".") for part in py_file.parts):
                continue
            python_files.append(py_file)
        return {
            "python_files": [str(file) for file in sorted(python_files)],
            "error": "",
        }
    except Exception as e:
        return {
            "python_files": [],
            "error": f"Error finding Python files: {str(e)}",
        }


@tool
def select_file_tool(python_files: List[str]) -> Dict[str, Any]:
    """Tool for selecting a Python file from a list of files.

    Args:
        python_files: List of Python file paths.

    Returns:
        Dict: Contains 'selected_file' and 'error'.
    """
    try:
        if not python_files:
            return {
                "selected_file": "",
                "error": "No Python files available for selection",
            }

        print("\n📁 Python files:")
        print("-" * 50)
        for i, file_path in enumerate(python_files, 1):
            try:
                print(f"{i:2d}. {Path(file_path).relative_to(Path.cwd())}")
            except ValueError:
                print(f"{i:2d}. {file_path}")
        print("-" * 50)

        while True:
            file_input = (
                input(
                    f"\nSelect file number (1-{len(python_files)}) "
                    "or q to quit: "
                )
                .strip()
                .lower()
            )

            if file_input == "q":
                return {
                    "selected_file": "",
                    "error": "File selection cancelled by user",
                }

            try:
                file_index = int(file_input) - 1
                if 0 <= file_index < len(python_files):
                    selected_file = python_files[file_index]
                    try:
                        rel_path = Path(selected_file).relative_to(Path.cwd())
                        print(f"\n📁 Selected file: {rel_path}")
                    except ValueError:
                        print(f"\n📁 Selected file: {selected_file}")
                    return {
                        "selected_file": selected_file,
                        "error": "",
                    }
                print(f"❌ Invalid file number. Use 1-{len(python_files)}.")
            except ValueError:
                print("❌ Please enter a valid number.")

    except KeyboardInterrupt:
        return {
            "selected_file": "",
            "error": "File selection cancelled by user",
        }
    except Exception as e:
        return {
            "selected_file": "",
            "error": f"Error selecting file: {str(e)}",
        }
