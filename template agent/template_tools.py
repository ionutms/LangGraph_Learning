from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def dummy_tool(input_data: str) -> Dict[str, Any]:
    """A dummy tool that processes input data.

    This tool doesn't perform any real processing - it's a placeholder
    that demonstrates the tool interface pattern. It simply formats the
    input with a basic processing indicator.

    Args:
        input_data: String containing the input data to "process".

    Returns:
        Dict: Contains 'processed_data' (the "processed" input) and 'error'.
    """
    try:
        # Validate inputs
        if not input_data:
            return {
                "processed_data": "",
                "error": "Input data cannot be empty",
            }

        # "Process" the data - just add a processing indicator
        processed = f"[PROCESSED] {input_data}"

        return {
            "processed_data": processed,
            "error": "",
        }

    except Exception as e:
        return {
            "processed_data": "",
            "error": f"Error in dummy tool processing: {str(e)}",
        }


@tool
def model_selection_tool(
    available_models: List[str], current_model: str = ""
) -> Dict[str, Any]:
    """Tool for selecting an LLM model from available options.

    Prompts the user to select a model from the list of available
    models and returns the selected model identifier.

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
                    print(f"\n🤖 Selected LLM Model: {selected_model}")
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
    """Tool for getting user input data.

    Prompts the user to enter input data and validates it.

    Returns:
        Dict: Contains 'input_data' and 'error'.
    """
    try:
        print("\n📝 Enter your input:")
        input_data = input("> ").strip()

        if not input_data:
            return {
                "input_data": "",
                "error": "Please provide some input.",
            }

        return {
            "input_data": input_data,
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
def display_tool(
    result: str, error: str = "", success: bool = True
) -> Dict[str, Any]:
    """Tool for displaying processing results to the user.

    Shows the processing results or error messages to the user in a
    formatted way.

    Args:
        result: The processing result to display.
        error: Error message if processing failed.
        success: Whether the processing was successful.

    Returns:
        Dict: Contains operation status and any error.
    """
    try:
        if success and result:
            print("\n📊 Result:")
            print(f"{result}")
            print("✨ Done!")
        elif error:
            print(f"❌ Error: {error}")
        else:
            print("⚠️  No result to display")

        return {
            "displayed": True,
            "error": "",
        }

    except Exception as e:
        return {
            "displayed": False,
            "error": f"Error displaying results: {str(e)}",
        }


@tool
def continue_prompt_tool() -> Dict[str, Any]:
    """Tool for asking user whether to continue processing.

    Prompts the user to decide whether to continue with more input
    or exit the application.

    Returns:
        Dict: Contains 'continue' (bool), 'user_input', and 'error'.
    """
    try:
        choice = input("\n🔄 Process more input? (y/n): ").lower().strip()

        continue_processing = choice in ["y", "yes"]

        return {
            "continue": continue_processing,
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
