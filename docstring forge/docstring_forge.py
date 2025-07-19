import ast
import sys
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("groq:llama-3.3-70b-versatile")

# Global LLM instructions for docstring operations
LLM_INSTRUCTIONS = {
    "update": """Improve the docstrings in the following Python code.
Make them more comprehensive, clear, and follow Google docstring
conventions without the example section.
Keep makimum 79 chars per line.
Remove whitespaces from generated docstrings.

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
""",
    "generate": """Please add comprehensive docstrings to all functions and
classes in the following Python code that are missing them.
Follow Google docstring conventions without the example section.
Keep makimum 79 chars per line.
Remove whitespaces from generated docstrings.
Preserve the original code functionality and structure.

Code:
```python
{original_code}
```

Please return the complete Python code with added docstrings.
For each function/class:
1. Add a clear description
2. Document all parameters with types
3. Document return values
4. Add examples for complex functions
5. Use consistent formatting
""",
}


# Define the state schema
class DocstringState(TypedDict):
    """State for the docstring processing graph.

    Attributes:
        file_path (str): Path to the Python file to process
        original_code (str): Original Python code content
        processed_code (str): Code after docstring processing
        action (str): Action to perform - 'remove', 'update', or 'generate'
        docstring_info (List[dict]): Information about found docstrings
        messages (list): Chat messages for LLM interaction
        error (Optional[str]): Error message if processing fails
        output_dir (str): Directory to save processed files
    """

    file_path: str
    original_code: str
    processed_code: str
    action: Literal["remove", "update", "generate"]
    docstring_info: List[dict]
    messages: Annotated[list, add_messages]
    error: Optional[str]
    output_dir: str


class DocstringProcessor:
    """Processes Python files to extract and manipulate docstrings."""

    def __init__(self):
        self.docstring_nodes = []

    def visit_function_or_class(self, node):
        """Extract docstring information from function or class nodes."""
        docstring_info = {
            "type": "function"
            if isinstance(node, ast.FunctionDef)
            else "class",
            "name": node.name,
            "lineno": node.lineno,
            "docstring": None,
            "docstring_node": None,
        }

        # Check if the first statement is a docstring
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring_info["docstring"] = node.body[0].value.value
            docstring_info["docstring_node"] = node.body[0]

        return docstring_info

    def extract_docstrings(self, code: str) -> List[dict]:
        """Extract all docstrings from Python code."""
        try:
            tree = ast.parse(code)
            docstrings = []

            # Check module-level docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                docstrings.append({
                    "type": "module",
                    "name": "__module__",
                    "lineno": tree.body[0].lineno,
                    "docstring": tree.body[0].value.value,
                    "docstring_node": tree.body[0],
                })

            # Walk through all nodes to find functions and classes
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                ):
                    docstring_info = self.visit_function_or_class(node)
                    if docstring_info["docstring"]:
                        docstrings.append(docstring_info)

            return docstrings
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

    def remove_docstrings(self, code: str) -> str:
        """Remove all docstrings from Python code."""
        try:
            # Parse the code to identify docstring locations
            tree = ast.parse(code)
            lines = code.split("\n")

            # Collect all docstring ranges to remove with context info
            docstring_ranges = []

            # Find module docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                start_line = tree.body[0].lineno - 1
                end_line = (
                    tree.body[0].end_lineno - 1
                    if tree.body[0].end_lineno
                    else start_line
                )
                docstring_ranges.append({
                    "start": start_line,
                    "end": end_line,
                    "type": "module",
                    "node": tree.body[0],
                })

            # Find function and class docstrings
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                ):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        docstring_node = node.body[0]
                        start_line = docstring_node.lineno - 1
                        end_line = (
                            docstring_node.end_lineno - 1
                            if docstring_node.end_lineno
                            else start_line
                        )
                        docstring_ranges.append({
                            "start": start_line,
                            "end": end_line,
                            "type": "function_class",
                            "node": docstring_node,
                            "parent": node,
                            "is_only_body": len(node.body) == 1,
                        })

            docstring_ranges.sort(key=lambda x: x["start"], reverse=True)

            # Process each docstring range
            for docstring_info in docstring_ranges:
                start_line = docstring_info["start"]
                end_line = docstring_info["end"]

                if start_line < len(lines):
                    indent_match = lines[start_line].lstrip()
                    indent_level = len(lines[start_line]) - len(indent_match)
                else:
                    indent_level = 0

                # Remove the docstring lines
                del lines[start_line : end_line + 1]

                # If this was the only content in a function/class, add 'pass'
                if docstring_info[
                    "type"
                ] == "function_class" and docstring_info.get(
                    "is_only_body", False
                ):
                    lines.insert(start_line, " " * indent_level + "pass")

            # Join the lines back together - preserve original formatting
            result = "\n".join(lines)

            # Only validate syntax without reformatting
            try:
                ast.parse(result)
            except SyntaxError as _e:
                # If there's a syntax error, try a minimal fix
                result = self._minimal_syntax_fix(result)

            return result

        except Exception as e:
            raise ValueError(f"Error removing docstrings: {e}")

    def _minimal_syntax_fix(self, code: str) -> str:
        """Apply minimal syntax fixes without reformatting the entire code."""
        try:
            return code  # If it parses, return as-is
        except SyntaxError:
            # Only if there's a real syntax issue, apply minimal fixes
            lines = code.split("\n")

            # Check for empty function/class bodies and add pass where needed
            in_function_or_class = False
            indent_stack = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue

                # Calculate indentation
                indent = len(line) - len(line.lstrip())

                # Check if this line defines a function or class
                if (
                    stripped.startswith("def ")
                    or stripped.startswith("async def ")
                    or stripped.startswith("class ")
                ):
                    in_function_or_class = True
                    indent_stack.append(indent)
                    continue

                if in_function_or_class and indent_stack:
                    expected_indent = indent_stack[-1]
                    if indent <= expected_indent:
                        j = i - 1
                        while j >= 0 and (
                            not lines[j].strip()
                            or lines[j].strip().startswith("#")
                        ):
                            j -= 1

                        if j >= 0:
                            prev_line = lines[j].strip()
                            if prev_line.endswith(":") and (
                                prev_line.startswith("def ")
                                or prev_line.startswith("async def ")
                                or prev_line.startswith("class ")
                            ):
                                # Insert pass statement
                                lines.insert(
                                    j + 1,
                                    " " * (expected_indent + 4) + "pass",
                                )

                        indent_stack.pop()
                        if not indent_stack:
                            in_function_or_class = False

            return "\n".join(lines)


class LLMPromptGenerator:
    """Generates prompts for LLM-based docstring operations."""

    @staticmethod
    def update_prompt(state: DocstringState) -> str:
        """Create prompt for updating existing docstrings."""
        docstrings_info = "\n".join([
            f"- {info['type']} '{info['name']}' (line {info['lineno']}): "
            f"{info['docstring'][:100]}..."
            for info in state["docstring_info"]
        ])

        return LLM_INSTRUCTIONS["update"].format(
            docstrings_info=docstrings_info,
            original_code=state["original_code"],
        )

    @staticmethod
    def generate_prompt(state: DocstringState) -> str:
        """Create prompt for generating missing docstrings."""
        return LLM_INSTRUCTIONS["generate"].format(
            original_code=state["original_code"]
        )


class FileManager:
    """Manages file operations and discovery."""

    @staticmethod
    def find_python_files(directory: Path) -> List[Path]:
        """Find all Python files in directory and subdirectories."""
        python_files = []

        # Skip common directories that usually don't need processing
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

        for py_file in directory.rglob("*.py"):
            # Skip files in excluded directories
            if any(part in skip_dirs for part in py_file.parts):
                continue
            # Skip hidden files
            if any(part.startswith(".") for part in py_file.parts):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    @staticmethod
    def load_file(file_path: Path) -> tuple[str, Optional[str]]:
        """Load and validate the Python file.

        Returns:
            tuple: (file_content, error_message) where error_message is
                   None on success
        """
        try:
            if not file_path.exists():
                return "", f"File not found: {file_path}"

            if not file_path.suffix == ".py":
                return "", f"File must be a Python file (.py): {file_path}"

            with open(file_path, "r", encoding="utf-8") as f:
                original_code = f.read()

            return original_code, None
        except Exception as e:
            return "", f"Error loading file: {e}"

    @staticmethod
    def save_file(
        file_path: Path, content: str, output_dir: Path
    ) -> Optional[str]:
        """Save the processed code to the output directory.

        Returns:
            Optional[str]: Error message if saving fails, None on success
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Preserve relative directory structure
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = file_path.name

            output_path = output_dir / relative_path

            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ Processed file saved: {output_path}")
            print(f"📁 Original file preserved: {file_path}")
            print(f"📂 Output directory: {output_dir}")

            return None

        except Exception as e:
            return f"Error saving file: {e}"


class UserInterface:
    """Handles user interaction and display."""

    @staticmethod
    def display_files_menu(files: List[Path]) -> None:
        """Display the list of available Python files."""
        print("📁 Available Python files:")
        print("-" * 50)
        for i, file_path in enumerate(files, 1):
            try:
                # Show file size and last modified info
                stat = file_path.stat()
                size = stat.st_size
                size_str = (
                    f"{size:,} bytes"
                    if size < 1024
                    else f"{size // 1024:,} KB"
                )
                print(
                    f"{i:2d}. {file_path.relative_to(Path.cwd())} "
                    f"({size_str})"
                )
            except OSError:
                print(f"{i:2d}. {file_path.relative_to(Path.cwd())}")
        print("-" * 50)

    @staticmethod
    def get_output_directory() -> str:
        """Get the output directory from user or use default."""
        print("\n📂 Output directory options:")
        print("  1. Use default: 'processed_files'")
        print("  2. Specify custom directory")

        while True:
            try:
                choice = input("Select option (1/2): ").strip()

                if choice == "1":
                    return "processed_files"
                elif choice == "2":
                    custom_dir = input(
                        "Enter output directory name: "
                    ).strip()
                    if custom_dir:
                        return custom_dir
                    else:
                        print("❌ Directory name cannot be empty.")
                else:
                    print("❌ Please choose 1 or 2.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)

    @staticmethod
    def get_user_choice(files: List[Path]) -> tuple[str, Path, str]:
        """Get user's choice of action, file, and output directory."""
        actions = {"r": "remove", "u": "update", "g": "generate"}

        print("\n🔧 Actions available:")
        print("  r - Remove all docstrings")
        print("  u - Update existing docstrings with LLM")
        print("  g - Generate missing docstrings with LLM")
        print("  q - Quit")

        while True:
            try:
                action_input = (
                    input("\nSelect action (r/u/g/q): ").lower().strip()
                )

                if action_input == "q":
                    print("👋 Goodbye!")
                    sys.exit(0)

                if action_input not in actions:
                    print("❌ Invalid action. Please choose r, u, g, or q.")
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
                        print(
                            f"❌ Invalid file number. Please choose "
                            f"1-{len(files)}."
                        )
                except ValueError:
                    print("❌ Please enter a valid number.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)


class GraphNodeHandler:
    """Handles the graph node operations."""

    def __init__(self, processor: DocstringProcessor):
        self.processor = processor
        self.prompt_generator = LLMPromptGenerator()
        self.file_manager = FileManager()

    def load_file(self, state: DocstringState) -> dict:
        """Load and validate the Python file."""
        file_path = Path(state["file_path"])
        original_code, error = self.file_manager.load_file(file_path)

        if error:
            return {"error": error}

        return {
            "original_code": original_code,
            "processed_code": original_code,
        }

    def analyze_docstrings(self, state: DocstringState) -> dict:
        """Analyze and extract docstring information from the code."""
        if state.get("error"):
            return {}

        try:
            docstring_info = self.processor.extract_docstrings(
                state["original_code"]
            )
            return {"docstring_info": docstring_info}
        except Exception as e:
            return {"error": f"Error analyzing docstrings: {e}"}

    def process_docstrings(self, state: DocstringState) -> dict:
        """Process docstrings based on the specified action."""
        if state.get("error"):
            return {}

        action = state["action"]

        try:
            if action == "remove":
                processed_code = self.processor.remove_docstrings(
                    state["original_code"]
                )
                return {"processed_code": processed_code}

            elif action == "update":
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": self.prompt_generator.update_prompt(
                                state
                            ),
                        }
                    ]
                }

            elif action == "generate":
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": self.prompt_generator.generate_prompt(
                                state
                            ),
                        }
                    ]
                }

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Error processing docstrings: {e}"}

    def llm_process(self, state: DocstringState) -> dict:
        """Use LLM to update or generate docstrings."""
        if state.get("error") or not state.get("messages"):
            return {}

        try:
            response = llm.invoke(state["messages"])

            # Extract the code from LLM response
            content = response.content

            # Try to extract code block
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
            return {"error": f"Error in LLM processing: {e}"}

    def save_result(self, state: DocstringState) -> dict:
        """Save the processed code to the output directory."""
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
    def should_use_llm(state: DocstringState) -> str:
        """Determine if LLM processing is needed."""
        if state.get("error"):
            return "save"
        return "llm" if state["action"] in ["update", "generate"] else "save"


class DocstringForge:
    """Main application class that orchestrates the processing workflow."""

    def __init__(self):
        self.processor = DocstringProcessor()
        self.handler = GraphNodeHandler(self.processor)
        self.ui = UserInterface()
        self.file_manager = FileManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the processing graph."""
        graph_builder = StateGraph(DocstringState)

        # Add nodes
        graph_builder.add_node("load", self.handler.load_file)
        graph_builder.add_node("analyze", self.handler.analyze_docstrings)
        graph_builder.add_node("process", self.handler.process_docstrings)
        graph_builder.add_node("llm", self.handler.llm_process)
        graph_builder.add_node("save", self.handler.save_result)

        # Add edges
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
        """Process a single file with the specified action."""
        print(f"\n🔧 Processing: {file_path.relative_to(Path.cwd())}")
        print(f"📝 Action: {action}")
        print(f"📂 Output directory: {output_dir}")
        print("-" * 50)

        # Initialize state
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
            # Run the graph
            result = self.graph.invoke(initial_state)

            if result.get("error"):
                print(f"❌ Error: {result['error']}")
                return

            # Show summary
            if action == "remove":
                print(
                    f"🗑️  Removed {len(result['docstring_info'])} docstrings"
                )
            else:
                print(
                    f"📚 Processed {len(result['docstring_info'])} existing "
                    "docstrings"
                )

            print("✨ Processing complete!")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def interactive_mode(self):
        """Run the docstring forge in interactive mode."""
        current_dir = Path.cwd()

        print("🔥 Docstring Forge - Interactive Mode")
        print(f"📂 Scanning directory: {current_dir}")
        print("=" * 60)

        # Find all Python files
        python_files = self.file_manager.find_python_files(current_dir)

        if not python_files:
            print("❌ No Python files found.")
            return

        print(f"✅ Found {len(python_files)} Python files")

        while True:
            try:
                self.ui.display_files_menu(python_files)
                action, selected_file, output_dir = self.ui.get_user_choice(
                    python_files
                )
                self.process_file(action, selected_file, output_dir)

                # Ask if user wants to process another file
                while True:
                    continue_choice = (
                        input("\n🔄 Process another file? (y/n): ")
                        .lower()
                        .strip()
                    )
                    if continue_choice in ["y", "yes"]:
                        break
                    elif continue_choice in ["n", "no"]:
                        print("👋 Goodbye!")
                        return
                    else:
                        print("❌ Please enter 'y' or 'n'.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return


if __name__ == "__main__":
    forge = DocstringForge()

    # Print the ASCII representation of the graph for debugging
    print("Graph structure:")
    print(forge.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    forge.interactive_mode()
