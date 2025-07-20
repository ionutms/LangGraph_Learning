import sys
from pathlib import Path
from typing import List

from docstring_forge_tools import (
    extract_docstrings_tool,
    load_file_tool,
    remove_docstrings_and_comments_tool,
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


class DocstringForgeHandlers:
    """Handlers for DocstringForge workflow steps using tools.

    Manages file operations, docstring analysis, processing, and saving for
    the docstring forge workflow.

    Attributes:
        llm: Initialized language model for docstring updates.
        sv_prompt: Prompt template for LLM docstring generation.
    """

    def __init__(self, llm, sv_prompt: str):
        """Initialize the handlers with LLM and prompt.

        Args:
            llm: Initialized language model for docstring updates.
            sv_prompt: Prompt template string for LLM docstring generation.
        """
        self.llm = llm
        self.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", sv_prompt),
            (
                "user",
                "{docstrings_info}\n\nOriginal code:"
                "\n```python\n{original_code}\n```",
            ),
        ])

    def load_file(self, state: dict) -> dict:
        """Load and validate the Python file using load_file_tool.

        Args:
            state: Agent state with file path.

        Returns:
            dict: Updated state with original and processed code, or error.
        """
        try:
            result = load_file_tool.invoke({"file_path": state["file_path"]})
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Error: {result['error']}")
                )
            else:
                state["original_code"] = result["file_content"]
                state["processed_code"] = result["file_content"]
                state["messages"].append(
                    AIMessage(content=f"Loaded file: {state['file_path']}")
                )
            return state
        except Exception as e:
            state["error"] = f"Error invoking load_file_tool: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def analyze_docstrings(self, state: dict) -> dict:
        """Analyze and extract docstring info using extract_docstrings_tool.

        Args:
            state: Agent state with original code.

        Returns:
            dict: Updated state with docstring info or error.
        """
        if state.get("error"):
            return state
        try:
            result = extract_docstrings_tool.invoke({
                "code": state["original_code"]
            })
            if result["error"]:
                state["error"] = result["error"]
                state["messages"].append(
                    AIMessage(content=f"Error: {result['error']}")
                )
            else:
                state["docstring_info"] = result["docstring_info"]
                state["messages"].append(
                    AIMessage(
                        content=f"Found {len(result['docstring_info'])} "
                        "docstrings"
                    )
                )
            return state
        except Exception as e:
            state["error"] = f"Error analyzing docstrings: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def process_docstrings(self, state: dict) -> dict:
        """Process docstrings and comments based on action using tools.

        Args:
            state: Agent state with action and original code.

        Returns:
            dict: Updated state with processed code or LLM messages, or error.
        """
        if state.get("error"):
            return state
        action = state["action"]
        try:
            if action == "remove":
                result = remove_docstrings_and_comments_tool.invoke({
                    "code": state["original_code"]
                })
                if result["error"]:
                    state["error"] = result["error"]
                    state["messages"].append(
                        AIMessage(content=f"Error: {result['error']}")
                    )
                else:
                    state["processed_code"] = result["processed_code"]
                    state["messages"].append(
                        AIMessage(content="Removed docstrings and comments")
                    )
                return state
            elif action == "update":
                docstrings_info = "\n".join([
                    f"- {info['type']} '{info['name']}' "
                    f"(line {info['lineno']}): "
                    f"{info['docstring'][:50]}..."
                    if info["docstring"]
                    else "None"
                    for info in state["docstring_info"]
                ])
                prompt = self.sv_prompt.invoke({
                    "docstrings_info": docstrings_info,
                    "original_code": state["original_code"],
                }).to_messages()
                state["messages"] = prompt
                state["messages"].append(
                    AIMessage(
                        content="Prepared LLM prompt for docstring update"
                    )
                )
                return state
            else:
                state["error"] = f"Unknown action: {action}"
                state["messages"].append(
                    AIMessage(content=f"Error: Unknown action {action}")
                )
                return state
        except Exception as e:
            state["error"] = f"Error processing docstrings: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def llm_process(self, state: dict) -> dict:
        """Use LLM to update docstrings.

        Args:
            state: Agent state with messages for LLM.

        Returns:
            dict: Updated state with processed code and messages, or error.
        """
        if state.get("error") or not state.get("messages"):
            return state
        try:
            response = self.llm.invoke(state["messages"])
            content = response.content
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                processed_code = (
                    content[start:end].strip()
                    if end != -1
                    else content[start:].strip()
                )
            else:
                processed_code = content.strip()
            state["processed_code"] = processed_code
            state["messages"].append(
                AIMessage(content="Updated docstrings with LLM")
            )
            return state
        except Exception as e:
            state["error"] = f"Error in LLM processing: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    def save_result(self, state: dict) -> dict:
        """Save the processed code to the output directory.

        Args:
            state: Agent state with processed code and output directory.

        Returns:
            dict: Updated state with saved file path or error.
        """
        if state.get("error"):
            return state
        try:
            output_dir = Path(state["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            original_path = Path(state["file_path"])
            output_path = output_dir / original_path.name
            content = state["processed_code"]
            if not content.endswith("\n"):
                content += "\n"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            state["saved_file"] = str(output_path)
            state["messages"].append(
                AIMessage(content=f"Saved processed file to {output_path}")
            )
            return state
        except Exception as e:
            state["error"] = f"Error saving file: {str(e)}"
            state["messages"].append(AIMessage(content=f"Error: {str(e)}"))
            return state

    @staticmethod
    def should_use_llm(state: dict) -> str:
        """Determine if LLM processing is needed.

        Args:
            state: Agent state with action and error status.

        Returns:
            str: Next edge to follow ('use_llm' or 'skip_llm').
        """
        if state.get("error"):
            return "skip_llm"
        return "use_llm" if state["action"] == "update" else "skip_llm"

    @staticmethod
    def display_files_menu(files: List[Path]) -> None:
        """Display the list of available Python files.

        Args:
            files: List of Path objects representing Python files.
        """
        print("📁 Python files:")
        print("-" * 50)
        for i, file_path in enumerate(files, 1):
            try:
                print(f"{i:2d}. {file_path.relative_to(Path.cwd())}")
            except OSError:
                print(f"{i:2d}. {file_path}")
        print("-" * 50)

    @staticmethod
    def get_output_directory() -> str:
        """Get the output directory from user or use default.

        Returns:
            str: Path to the output directory.
        """
        return "processed_files"

    @staticmethod
    def get_user_choice(files: List[Path]) -> tuple[str, Path, str]:
        """Get user's choice of action, file, and output directory.

        Args:
            files: List of Path objects representing Python files.

        Returns:
            tuple: (action, selected_file, output_dir).
        """
        actions = {"r": "remove", "u": "update"}
        print("\n🔧 Actions:")
        print("  r - Remove docstrings/comments")
        print("  u - Update docstrings with LLM")
        print("  q - Quit")
        while True:
            try:
                action_input = input("\nSelect action: ").lower().strip()
                if action_input == "q":
                    print("👋 Goodbye!")
                    sys.exit(0)
                if action_input not in actions:
                    print("❌ Invalid action. Use r, u, or q.")
                    continue
                file_input = input(
                    f"Select file number (1-{len(files)}): "
                ).strip()
                try:
                    file_index = int(file_input) - 1
                    if 0 <= file_index < len(files):
                        return (
                            actions[action_input],
                            files[file_index],
                            DocstringForgeHandlers.get_output_directory(),
                        )
                    print(f"❌ Invalid file number. Use 1-{len(files)}.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
