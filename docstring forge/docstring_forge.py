import ast
import sys
import tokenize
from io import StringIO
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
)

# Global LLM instructions for docstring operations
LLM_INSTRUCTIONS = {
    "update": """
Improve the docstrings in the following Python code.
Make them more comprehensive, clear, and follow Google docstring
conventions without the example section.
Keep maximum 79 chars per line.
Remove whitespaces from generated docstrings at lines end.
Add docstrings to all functions and classes that do not have them.
Don't make other changes to the provided code.

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
}


class DocstringState(TypedDict):
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
    action: Literal["remove", "update"]
    docstring_info: List[dict]
    messages: Annotated[list, add_messages]
    error: Optional[str]
    output_dir: str


class DocstringProcessor:
    """Processes Python files to manage docstrings and comments."""

    def __init__(self):
        self.docstring_nodes = []

    def visit_function_or_class(self, node):
        """Extract docstring info from function or class nodes.

        Args:
            node: AST node for a function or class definition.

        Returns:
            dict: Information about the node's docstring.
        """
        docstring_info = {
            "type": "function"
            if isinstance(node, ast.FunctionDef)
            else "class",
            "name": node.name,
            "lineno": node.lineno,
            "docstring": None,
            "docstring_node": None,
        }

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
        """Extract all docstrings from Python code.

        Args:
            code: String containing the Python code to analyze.

        Returns:
            List[dict]: List of dictionaries with docstring info.

        Raises:
            ValueError: If the code contains invalid Python syntax.
        """
        try:
            tree = ast.parse(code)
            docstrings = []

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

    def remove_docstrings_and_comments(self, code: str) -> str:
        """Remove all docstrings and comments from Python code.

        Args:
            code: String containing the Python code to process.

        Returns:
            str: Code with docstrings and comments removed.

        Raises:
            ValueError: If an error occurs during processing.
        """
        try:
            tree = ast.parse(code)
            lines = code.split("\n")
            docstring_ranges = []
            comment_lines = set()

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

            # Find all comments using tokenize
            string_io = StringIO(code)
            tokens = list(tokenize.generate_tokens(string_io.readline))
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    start_line = token.start[0] - 1
                    comment_lines.add(start_line)

            # Combine and sort ranges
            all_ranges = docstring_ranges + [
                {"start": line, "end": line, "type": "comment"}
                for line in comment_lines
            ]
            all_ranges.sort(key=lambda x: x["start"], reverse=True)

            # Process each range
            for range_info in all_ranges:
                start_line = range_info["start"]
                end_line = range_info["end"]

                if start_line < len(lines):
                    indent_match = lines[start_line].lstrip()
                    indent_level = len(lines[start_line]) - len(indent_match)
                else:
                    indent_level = 0

                del lines[start_line : end_line + 1]

                if range_info["type"] == "function_class" and range_info.get(
                    "is_only_body", False
                ):
                    lines.insert(start_line, " " * indent_level + "pass")

            result = "\n".join(lines)

            try:
                ast.parse(result)
            except SyntaxError:
                result = self._minimal_syntax_fix(result)

            return result

        except Exception as e:
            raise ValueError(f"Error removing docstrings and comments: {e}")

    def _minimal_syntax_fix(self, code: str) -> str:
        """Apply minimal syntax fixes without reformatting code.

        Args:
            code: String containing the Python code to fix.

        Returns:
            str: Code with minimal syntax fixes applied.
        """
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            lines = code.split("\n")
            in_function_or_class = False
            indent_stack = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue

                indent = len(line) - len(line.lstrip())

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
                        while j >= 0 and not lines[j].strip():
                            j -= 1

                        if j >= 0:
                            prev_line = lines[j].strip()
                            if prev_line.endswith(":") and (
                                prev_line.startswith("def ")
                                or prev_line.startswith("async def ")
                                or prev_line.startswith("class ")
                            ):
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

        return LLM_INSTRUCTIONS["update"].format(
            docstrings_info=docstrings_info,
            original_code=state["original_code"],
        )


class FileManager:
    """Manages file operations and discovery."""

    @staticmethod
    def find_python_files(directory: Path) -> List[Path]:
        """Find all Python files in directory and subdirectories.

        Args:
            directory: Path object for the directory to search.

        Returns:
            List[Path]: Sorted list of Python file paths.
        """
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

        for py_file in directory.rglob("*.py"):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            if any(part.startswith(".") for part in py_file.parts):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    @staticmethod
    def load_file(file_path: Path) -> tuple[str, Optional[str]]:
        """Load and validate the Python file.

        Args:
            file_path: Path object for the file to load.

        Returns:
            tuple: (file_content, error_message) where error_message
                   is None on success.
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

        Args:
            file_path: Original file path.
            content: Processed code content to save.
            output_dir: Directory to save the processed file.

        Returns:
            Optional[str]: Error message if saving fails, None on success.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = file_path.name

            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not content.endswith("\n"):
                content += "\n"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ Processed file saved: {output_path}")
            print(f"📁 Original file preserved: {file_path}")
            print(f"📂 Output directory: {output_dir}")

        except Exception as e:
            return f"Error saving file: {e}"


class UserInterface:
    """Handles user interaction and display."""

    @staticmethod
    def display_files_menu(files: List[Path]) -> None:
        """Display the list of available Python files.

        Args:
            files: List of Path objects representing Python files.
        """
        print("📁 Available Python files:")
        print("-" * 50)
        for i, file_path in enumerate(files, 1):
            try:
                stat = file_path.stat()
                size = stat.st_size
                size_str = (
                    f"{size:,} bytes"
                    if size < 1024
                    else f"{size // 1024:,} KB"
                )
                print(
                    f"{i:2d}. {file_path.relative_to(Path.cwd())} ({size_str})"
                )
            except OSError:
                print(f"{i:2d}. {file_path.relative_to(Path.cwd())}")
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
        print("\n🔧 Actions available:")
        print("  r - Remove all docstrings and comments")
        print("  u - Update existing docstrings with LLM")
        print("  q - Quit")

        while True:
            try:
                action_input = (
                    input("\nSelect action (r/u/q): ").lower().strip()
                )

                if action_input == "q":
                    print("👋 Goodbye!")
                    sys.exit(0)

                if action_input not in actions:
                    print("❌ Invalid action. Please choose r, u, or q.")
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
                            f"❌ Invalid file number. Please choose 1-{len(files)}."
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
        """Load and validate the Python file.

        Args:
            state: The current state containing file path.

        Returns:
            dict: Updated state with original and processed code.
        """
        file_path = Path(state["file_path"])
        original_code, error = self.file_manager.load_file(file_path)

        if error:
            return {"error": error}

        return {
            "original_code": original_code,
            "processed_code": original_code,
        }

    def analyze_docstrings(self, state: DocstringState) -> dict:
        """Analyze and extract docstring info from the code.

        Args:
            state: The current state containing the original code.

        Returns:
            dict: Updated state with docstring information.
        """
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
        """Process docstrings and comments based on the action.

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
                processed_code = (
                    self.processor.remove_docstrings_and_comments(
                        state["original_code"]
                    )
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

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Error processing docstrings: {e}"}

    def llm_process(self, state: DocstringState) -> dict:
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
            return {"error": f"Error in LLM processing: {e}"}

    def save_result(self, state: DocstringState) -> dict:
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
    def should_use_llm(state: DocstringState) -> str:
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
        graph_builder = StateGraph(DocstringState)
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
        print(f"\n🔧 Processing: {file_path.relative_to(Path.cwd())}")
        print(f"📝 Action: {action}")
        print(f"📂 Output directory: {output_dir}")
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
                print(
                    f"🗑️ Removed {len(result['docstring_info'])} docstrings and comments"
                )
            else:
                print(
                    f"📚 Processed {len(result['docstring_info'])} existing docstrings"
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
    print("Graph structure:")
    print(forge.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")
    forge.interactive_mode()
