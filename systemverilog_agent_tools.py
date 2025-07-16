import os
import re
import shutil
from typing import Any, Dict

from langchain_core.tools import tool

from verilator_runner import run_docker_compose


@tool
def generate_env_file_tool(
    generated_code: str, output_dir: str = "output", user_request: str = ""
) -> Dict[str, Any]:
    """Generates .env file content for a SystemVerilog project.

    Args:
        generated_code: Design code to extract module name from.
        output_dir: Base directory for saving files (default: 'output').
        user_request: Original user request to store in .env.

    Returns:
        Dict: Contains env_content and any error message.

    Raises:
        Exception: If module name extraction or .env generation fails.
    """
    try:
        # Extract module name from design code
        module_match = re.search(r"module\s+(\w+)", generated_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )

        # Create relative path to the module directory
        module_relative_path = f"./{output_dir}/{module_name}"

        design_filename = f"{module_name}.sv"
        testbench_filename = f"{module_name}_tb.sv"

        # Define .env key-value pairs with relative paths
        env_lines = [
            f"PROJECT_DIR={module_relative_path}",
            f"DESIGN_FILE={design_filename}",
            f"TESTBENCH_FILE={testbench_filename}",
            f"TOP_MODULE={module_name}_tb",
            f"VCD_FILE={module_name}_tb.vcd",
            f"USER_REQUEST={user_request}",
        ]

        # Join lines with newlines for .env content
        env_content = "\n".join(env_lines)

        return {"env_content": env_content, "error": ""}
    except Exception as error:
        return {
            "env_content": "",
            "error": f"Environment file generation error: {str(error)}",
        }


@tool
def save_code_tool(
    design_code: str,
    testbench_code: str,
    env_content: str,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Saves SystemVerilog code and .env to a module-specific directory.

    Args:
        design_code: Generated SystemVerilog design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        output_dir: Base directory for saving files (default: 'output').

    Returns:
        Dict: Contains status messages, saved file paths, module_dir,
            module_name, and error message.

    Raises:
        OSError: If directory creation or file writing fails.
        Exception: For unexpected errors during file saving.
    """
    try:
        messages = []
        saved_files = {}

        module_match = re.search(r"module\s+(\w+)", design_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )

        # Create module-specific directory
        module_dir = os.path.join(output_dir, module_name)
        os.makedirs(module_dir, exist_ok=True)

        # Save design code
        design_filename = f"{module_name}.sv"
        design_filepath = os.path.join(module_dir, design_filename)
        with open(design_filepath, "w") as design_file:
            design_file.write(design_code)
        messages.append(f"Design saved to {design_filepath}")
        saved_files["design_file"] = design_filepath

        # Save testbench code if available
        testbench_filename = f"{module_name}_tb.sv"
        testbench_filepath = os.path.join(module_dir, testbench_filename)
        if testbench_code:
            with open(testbench_filepath, "w") as testbench_file:
                testbench_file.write(testbench_code)
            messages.append(f"Testbench saved to {testbench_filepath}")
            saved_files["testbench_file"] = testbench_filepath
        else:
            messages.append("No testbench code to save")

        # Save .env file
        env_filepath = os.path.join(module_dir, ".env")
        if env_content:
            with open(env_filepath, "w") as env_file:
                env_file.write(env_content)
            messages.append(f"Environment file saved to {env_filepath}")
            saved_files["env_file"] = env_filepath
        else:
            messages.append("No .env file content to save")

        return {
            "messages": messages,
            "error": "",
            "module_dir": module_dir,
            "saved_files": saved_files,
            "module_name": module_name,
        }
    except Exception as error:
        return {
            "messages": [],
            "error": f"Error saving code: {str(error)}",
            "module_dir": "",
            "saved_files": {},
            "module_name": "",
        }


@tool
def run_simulation_tool(
    target_dir: str, strip_lines: bool = True
) -> Dict[str, Any]:
    """Runs Verilator simulation using Docker Compose.

    Args:
        target_dir: Directory with .env and SystemVerilog files.
        strip_lines: Strips first/last lines from output (default: True).

    Returns:
        Dict: Contains success status, return code, message, and error.

    Raises:
        Exception: If simulation execution fails.
    """
    try:
        # Ensure target directory ends with separator for run_docker_compose
        if not target_dir.endswith(os.sep):
            target_dir += os.sep

        print(f"Starting Verilator simulation in: {target_dir}")

        # Run the simulation
        return_code = run_docker_compose(
            target_dir=target_dir, strip_lines=strip_lines
        )

        success = return_code == 0

        if success:
            message = "Verilator simulation completed successfully"
        else:
            message = (
                f"Verilator simulation failed with return code: {return_code}"
            )

        return {
            "success": success,
            "return_code": return_code,
            "message": message,
            "error": ""
            if success
            else f"Simulation failed (exit code: {return_code})",
        }

    except Exception as error:
        return {
            "success": False,
            "return_code": -1,
            "message": f"Error running simulation: {str(error)}",
            "error": f"Simulation tool error: {str(error)}",
        }


@tool
def cleanup_files_tool(
    module_dir: str, saved_files: Dict[str, str]
) -> Dict[str, Any]:
    """Remove generated files and directory when simulation fails.

    This tool cleans up all generated files and the module directory
    when a simulation fails, helping to maintain a clean workspace.

    Args:
        module_dir: Path to the module directory to remove
        saved_files: Dictionary of saved files to remove (for verification)

    Returns:
        Dict containing:
            - success: Boolean indicating if cleanup was successful
            - message: Status message about cleanup operation
            - error: Error message if cleanup failed
            - removed_files: List of files that were removed
            - removed_dir: Directory that was removed
    """
    try:
        removed_files = []
        removed_dir = ""

        # Check if module directory exists
        if not os.path.exists(module_dir):
            return {
                "success": True,
                "message": f"Directory {module_dir} does not exist",
                "error": "",
                "removed_files": [],
                "removed_dir": "",
            }

        # List files that will be removed for logging
        if os.path.isdir(module_dir):
            for root, dirs, files in os.walk(module_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    removed_files.append(file_path)

        # Remove the entire module directory and its contents
        shutil.rmtree(module_dir)
        removed_dir = module_dir

        message = (
            f"Successfully cleaned up {len(removed_files)} files and ",
            f"removed directory: {module_dir}",
        )

        return {
            "success": True,
            "message": message,
            "error": "",
            "removed_files": removed_files,
            "removed_dir": removed_dir,
        }

    except PermissionError as e:
        error_msg = (
            f"Permission denied while cleaning up {module_dir}: {str(e)}"
        )
        return {
            "success": False,
            "message": "",
            "error": error_msg,
            "removed_files": [],
            "removed_dir": "",
        }
    except FileNotFoundError as e:
        # This shouldn't happen since we check existence, but handle it anyway
        return {
            "success": True,
            "message": f"Files already removed or not found: {str(e)}",
            "error": "",
            "removed_files": [],
            "removed_dir": "",
        }
    except Exception as e:
        error_msg = f"Error during cleanup of {module_dir}: {str(e)}"
        return {
            "success": False,
            "message": "",
            "error": error_msg,
            "removed_files": [],
            "removed_dir": "",
        }
