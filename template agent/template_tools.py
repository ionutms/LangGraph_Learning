from typing import Any, Dict

from langchain_core.tools import tool


@tool
def dummy_tool(input_data: str) -> Dict[str, Any]:
    """A dummy tool that processes input data.

    This tool doesn't perform any real processing - it's a placeholder that
    demonstrates the tool interface pattern. It simply formats the input
    with a basic processing indicator.

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
