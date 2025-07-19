import sys
from pathlib import Path
from typing import Annotated, List, Optional

from docstring_forge_agent_tools import (
    extract_docstrings_tool,
    find_python_files_tool,
    load_file_tool,
    remove_docstrings_and_comments_tool,
)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=12000,
)

# Global LLM instructions for docstring operations
UPDATE_PROMPT = """Improve the docstrings in the following Python code.
Make them more comprehensive, clear, and follow Google docstring
conventions without the example section.
Keep maximum 79 chars per line.
Remove whitespaces from generated docstrings at lines end.
Add docstrings to all functions and classes that do not have them.
Don't make other changes to the provided code.
Don't refactor the code.

Current docstrings found:
{docstrings_info}

Original code:
```python
{original_code}
```

Please return the complete updated Python code with improved docstrings.
Focus on:
1. Clear descriptions
2. Proper parameter documentation
3. Return value documentation
4. Example usage where appropriate
5. Consistent formatting
"""


class AgentState(TypedDict):
    """State for docstring and comment processing graph.

    Attributes:
        file_path: Path to the Python file to process.
        original_code: Original Python code content.
        processed_code: Code after docstring and comment processing.
        action: Action to perform - 'remove' or 'update'.
        docstring_info: Information about found docstrings.
        messages: Chat messages for LLM interaction.
        error: Error message if processing fails, None otherwise.
        output_dir: Directory to save processed files.
    """

    file_path: str
    original_code: str
    processed_code: str
    action: str
    docstring_info: List[dict]
    messages: Annotated[list, add_messages]
    error: Optional[str]
    output_dir: str


class DocstringProcessor:
    """Processes Python files to manage docstrings and comments."""

    def __init__(self):
        self.docstring_nodes = []


class LLMPromptGenerator:
    """Generates prompts for LLM-based docstring operations."""

    @staticmethod
    def update_prompt(state: AgentState) -> str:
        """Create prompt for updating existing docstrings.

        Args:
            state: The current state with docstring info and code.

        Returns:
            str: Formatted prompt for the LLM.
        """
        docstrings_info = "\n".join([
            f"- {info['type']} '{info['name']}' (line {info['lineno']}): "
            f"{info['docstring'][:50]}..."
            for info in state["docstring_info"]
        ])

        return UPDATE_PROMPT.format(
            docstrings_info=docstrings_info,
            original_code=state["original_code"],
        )


class FileManager:
    """Manages file operations and discovery."""

    @staticmethod
    def save_file(
        file_path: Path, content: str, output_dir: Path
    ) -> Optional[str]:
        """Save the processed code to the output directory.

        Args:
            file_path: Original file path.
            content: Processed code content to save.
            output_dir: Directory to save the processed file.

        Returns:
            Optional[str]: Error message if saving fails, None on success.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use only the filename from the original path
            output_path = output_dir / file_path.name

            if not content.endswith("\n"):
                content += "\n"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ Saved: {output_path}")
            print(f"📁 Original: {file_path}")
            print(f"📂 Output: {output_dir}")

        except Exception as e:
            return f"Error saving file: {str(e)}"


class UserInterface:
    """Handles user interaction and display."""

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
                        output_dir = UserInterface.get_output_directory()
                        return (
                            actions[action_input],
                            files[file_index],
                            output_dir,
                        )
                    else:
                        print(f"❌ Invalid file number. Use 1-{len(files)}.")
                except ValueError:
                    print("❌ Please enter a valid number.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)


class GraphNodeHandler:
    """Handles the graph node operations for docstring processing."""

    def __init__(self, processor: DocstringProcessor):
        self.processor = processor
        self.prompt_generator = LLMPromptGenerator()
        self.file_manager = FileManager()

    def load_file(self, state: AgentState) -> dict:
        """Load and validate the Python file using load_file_tool.

        Args:
            state: The current state containing file path.

        Returns:
            dict: Updated state with original and processed code.
        """
        try:
            result = load_file_tool.invoke({"file_path": state["file_path"]})
            if result["error"]:
                return {"error": result["error"]}
            return {
                "original_code": result["file_content"],
                "processed_code": result["file_content"],
            }
        except Exception as e:
            return {"error": f"Error invoking load_file_tool: {str(e)}"}

    def analyze_docstrings(self, state: AgentState) -> dict:
        """Analyze and extract docstring info using extract_docstrings_tool.

        Args:
            state: The current state containing the original code.

        Returns:
            dict: Updated state with docstring information.
        """
        if state.get("error"):
            return {}

        try:
            result = extract_docstrings_tool.invoke({
                "code": state["original_code"]
            })
            if result["error"]:
                return {"error": result["error"]}
            return {"docstring_info": result["docstring_info"]}
        except Exception as e:
            return {
                "error": f"Error invoking extract_docstrings_tool: {str(e)}"
            }

    def process_docstrings(self, state: AgentState) -> dict:
        """Process docstrings and comments based on action using tools.

        Args:
            state: The current state containing action and code.

        Returns:
            dict: Updated state with processed code or LLM messages.
        """
        if state.get("error"):
            return {}

        action = state["action"]

        try:
            if action == "remove":
                result = remove_docstrings_and_comments_tool.invoke({
                    "code": state["original_code"]
                })
                if result["error"]:
                    return {"error": result["error"]}
                return {"processed_code": result["processed_code"]}

            elif action == "update":
                prompt = self.prompt_generator.update_prompt(state)
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Error processing docstrings: {str(e)}"}

    def llm_process(self, state: AgentState) -> dict:
        """Use LLM to update docstrings.

        Args:
            state: The current state containing messages.

        Returns:
            dict: Updated state with processed code and messages.
        """
        if state.get("error") or not state.get("messages"):
            return {}

        try:
            response = llm.invoke(state["messages"])
            content = response.content

            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                if end != -1:
                    processed_code = content[start:end].strip()
                else:
                    processed_code = content[start:].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    processed_code = content[start:end].strip()
                else:
                    processed_code = content[start:].strip()
            else:
                processed_code = content.strip()

            return {
                "processed_code": processed_code,
                "messages": state["messages"] + [response],
            }

        except Exception as e:
            return {"error": f"Error in LLM processing: {str(e)}"}

    def save_result(self, state: AgentState) -> dict:
        """Save the processed code to the output directory.

        Args:
            state: The current state containing processed code.

        Returns:
            dict: Updated state with error info if any.
        """
        if state.get("error"):
            return {}

        original_path = Path(state["file_path"])
        output_dir = Path(state["output_dir"])

        error = self.file_manager.save_file(
            original_path, state["processed_code"], output_dir
        )

        if error:
            return {"error": error}

        return {}

    @staticmethod
    def should_use_llm(state: AgentState) -> str:
        """Determine if LLM processing is needed.

        Args:
            state: The current state with action and error status.

        Returns:
            str: Next node to process ('llm' or 'save').
        """
        if state.get("error"):
            return "save"
        return "llm" if state["action"] == "update" else "save"


class DocstringForge:
    """Main class that orchestrates the processing workflow."""

    def __init__(self):
        self.processor = DocstringProcessor()
        self.handler = GraphNodeHandler(self.processor)
        self.ui = UserInterface()
        self.file_manager = FileManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the processing graph.

        Returns:
            StateGraph: Compiled graph for processing docstrings
                        and comments.
        """
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("load", self.handler.load_file)
        graph_builder.add_node("analyze", self.handler.analyze_docstrings)
        graph_builder.add_node("process", self.handler.process_docstrings)
        graph_builder.add_node("llm", self.handler.llm_process)
        graph_builder.add_node("save", self.handler.save_result)
        graph_builder.add_edge(START, "load")
        graph_builder.add_edge("load", "analyze")
        graph_builder.add_edge("analyze", "process")
        graph_builder.add_conditional_edges(
            "process",
            self.handler.should_use_llm,
            {"llm": "llm", "save": "save"},
        )
        graph_builder.add_edge("llm", "save")
        graph_builder.add_edge("save", END)
        return graph_builder.compile()

    def process_file(
        self, action: str, file_path: Path, output_dir: str
    ) -> None:
        """Process a single file with the specified action.

        Args:
            action: Action to perform ('remove' or 'update').
            file_path: Path to the Python file to process.
            output_dir: Directory to save the processed file.
        """
        rel_path = file_path.relative_to(Path.cwd())
        print(f"🔧 Processing: {rel_path}")
        print(f"📝 Action: {action}")
        print(f"📂 Output: {output_dir}")
        print("-" * 50)

        initial_state = {
            "file_path": str(file_path),
            "action": action,
            "original_code": "",
            "processed_code": "",
            "docstring_info": [],
            "messages": [],
            "error": None,
            "output_dir": output_dir,
        }

        try:
            result = self.graph.invoke(initial_state)

            if result.get("error"):
                print(f"❌ Error: {result['error']}")
                return

            if action == "remove":
                print(f"🗑️ Removed {len(result['docstring_info'])} docstrings")
            else:
                print(
                    f"📚 Updated {len(result['docstring_info'])} docstrings"
                )

            print("✨ Done!")

        except Exception as e:
            print(f"❌ Error: {e}")

    def interactive_mode(self):
        """Run the docstring forge in interactive mode."""
        current_dir = Path.cwd()
        print("🔥 Docstring Forge - Interactive")
        print(f"📂 Scanning: {current_dir}")
        print("=" * 60)

        while True:
            try:
                result = find_python_files_tool.invoke({
                    "directory": str(current_dir)
                })
                if result["error"]:
                    print(f"❌ Error: {result['error']}")
                    return

                python_files = [Path(f) for f in result["python_files"]]

                if not python_files:
                    print("❌ No Python files found.")
                    return

                print(f"✅ Found {len(python_files)} Python files")

                self.ui.display_files_menu(python_files)
                action, selected_file, output_dir = self.ui.get_user_choice(
                    python_files
                )

                max_repeats = 5
                repeat_count = 0

                while repeat_count < max_repeats:
                    self.process_file(action, selected_file, output_dir)
                    repeat_count += 1

                    if repeat_count < max_repeats:
                        print(f"🔄 Repeated {repeat_count} of {max_repeats}")
                        choice = (
                            input(
                                f"🔄 Repeat on {selected_file.name}? (y/n): "
                            )
                            .lower()
                            .strip()
                        )
                        if choice not in ["y", "yes"]:
                            break
                    else:
                        print(f"🔄 Max repeats ({max_repeats}) reached")

                while True:
                    choice = input("\n🔄 Process another file? (y/n): ")
                    choice = choice.lower().strip()
                    if choice in ["y", "yes"]:
                        break
                    elif choice in ["n", "no"]:
                        print("👋 Goodbye!")
                        return
                    else:
                        print("❌ Enter 'y' or 'n'.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    forge = DocstringForge()
    print("Graph structure:")
    print(forge.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")
    forge.interactive_mode()
