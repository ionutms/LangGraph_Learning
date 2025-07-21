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
def user_input_tool() -> Dict[str, Any]:
    """Tool for getting user input for chat.

    Returns:
        Dict: Contains 'input_data' and 'error'.
    """
    try:
        print("\n💬 You:")
        user_message = input("> ").strip()

        if not user_message:
            return {
                "input_data": "",
                "error": "Please enter a message.",
            }

        return {
            "input_data": user_message,
            "error": "",
        }

    except KeyboardInterrupt:
        return {
            "input_data": "",
            "error": "Input cancelled by user",
        }
    except Exception as e:
        return {
            "input_data": "",
            "error": f"Error getting user input: {str(e)}",
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
