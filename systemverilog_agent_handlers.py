"""SystemVerilog agent handlers module.

This module contains handler methods that use tools for the SystemVerilog
code generation workflow. Separated from the main agent class to improve
code organization and maintainability.
"""

import re
from typing import Any, Dict

from langchain_core.messages import AIMessage

from systemverilog_agent_tools import (
    cleanup_files_tool,
    generate_env_file_tool,
    run_simulation_tool,
    save_code_tool,
)


class SVHandlers:
    """Handlers for SystemVerilog agent workflow steps that use tools."""

    @staticmethod
    def create_env_file(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generates .env file content using generate_env_file_tool.

        Uses the LLM-generated path to create .env with project config.

        Args:
            state: Agent state with generated code and path.

        Returns:
            Dict: Updated state with .env content and status messages.

        Raises:
            Exception: If .env file generation fails.
        """
        try:
            result = generate_env_file_tool.invoke({
                "generated_code": state["generated_code"],
                "output_dir": state["generated_path"],
            })
            state["env_content"] = result["env_content"]
            if result["error"]:
                state["error"] = result["error"]
            else:
                module_name = (
                    re.search(
                        r"module\s+(\w+)", state["generated_code"]
                    ).group(1)
                    if re.search(r"module\s+(\w+)", state["generated_code"])
                    else "generated_module"
                )
                state["messages"].append(
                    AIMessage(
                        content=(
                            f"Generated .env file for {module_name} in "
                            f"{state['generated_path']}"
                        )
                    )
                )
        except Exception as error:
            state["error"] = f"Error generating .env file: {str(error)}"
        return state

    @staticmethod
    def save_generated_files(state: Dict[str, Any]) -> Dict[str, Any]:
        """Saves generated code and .env using save_code_tool.

        Stores design, testbench, and .env files in the LLM-generated path.
        If module_dir already exists (from regeneration), uses the same
        directory.

        Args:
            state: Agent state with code, testbench, and .env content.

        Returns:
            Dict: Updated state with save status messages and file paths.

        Raises:
            Exception: If file saving fails due to I/O or other errors.
        """
        try:
            # Pass existing module_dir if available (for regeneration)
            save_params = {
                "design_code": state["generated_code"],
                "testbench_code": state["testbench_code"],
                "env_content": state["env_content"],
                "output_dir": state["generated_path"],
            }

            # If we have an existing module_dir, use it for regeneration
            if state.get("module_dir"):
                save_params["existing_module_dir"] = state["module_dir"]

            result = save_code_tool.invoke(save_params)

            for message in result["messages"]:
                state["messages"].append(AIMessage(content=message))
            if result["error"]:
                state["error"] = result["error"]
            else:
                # Store the module directory and saved files for later use
                state["module_dir"] = result.get("module_dir", "")
                state["saved_files"] = result.get("saved_files", {})
        except Exception as error:
            state["error"] = f"Error saving files: {str(error)}"
        return state

    @staticmethod
    def execute_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes Verilator simulation using run_simulation_tool.

        Runs simulation for saved SystemVerilog files in module directory.

        Args:
            state: Agent state with module directory and other data.

        Returns:
            Dict: Updated state with simulation results and messages.

        Raises:
            Exception: If simulation execution fails.
        """
        try:
            if not state["module_dir"]:
                state["error"] = "Module directory not set for simulation"
                state["messages"].append(
                    AIMessage(content="Error: Module directory not set")
                )
                return state
            result = run_simulation_tool.invoke({
                "target_dir": state["module_dir"],
                "strip_lines": True,
            })
            state["messages"].append(AIMessage(content=result["message"]))
            if result["error"]:
                state["error"] = result["error"]
        except Exception as error:
            state["error"] = f"Error executing simulation: {str(error)}"
        return state

    @staticmethod
    def cleanup_on_failure(state: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans up generated files when simulation fails.

        Removes all generated files and the module directory to maintain
        a clean workspace before offering retry option.

        Args:
            state: Agent state with module directory and saved files info.

        Returns:
            Dict: Updated state with cleanup status and messages.
        """
        try:
            if not state["module_dir"]:
                state["messages"].append(
                    AIMessage(content="No module directory to clean up")
                )
                state["cleanup_performed"] = False
                return state

            result = cleanup_files_tool.invoke({
                "module_dir": state["module_dir"],
                "saved_files": state["saved_files"],
            })

            if result["success"]:
                state["messages"].append(
                    AIMessage(content=f"🧹 Cleanup: {result['message']}")
                )
                state["cleanup_performed"] = True
                # Clear the module_dir and saved_files since they're gone
                state["module_dir"] = ""
                state["saved_files"] = {}
            else:
                state["messages"].append(
                    AIMessage(content=f"❌ Cleanup failed: {result['error']}")
                )
                state["cleanup_performed"] = False
                # Don't clear module_dir if cleanup failed

        except Exception as error:
            state["error"] = f"Error during cleanup: {str(error)}"
            state["messages"].append(
                AIMessage(content=f"❌ Cleanup error: {str(error)}")
            )
            state["cleanup_performed"] = False

        return state
