import ast
import tokenize
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def model_selection_tool(
    models: List[str], current: str = ""
) -> Dict[str, Any]:
    """Select an LLM model from available options.

    Args:
        models: List of model identifiers.
        current: Currently selected model (if any).

    Returns:
        Dict: Contains 'selected_model' and 'error'.
    """
    try:
        if not models:
            return {"selected_model": "", "error": "No models available"}
        print("\nAvailable LLM Models:")
        for i, model in enumerate(models, 1):
            mark = " (current)" if model == current else ""
            print(f"{i}. {model}{mark}")
        while True:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            try:
                file_index = int(choice) - 1
                if 0 <= file_index < len(models):
                    selected = models[file_index]
                    print(f"\n🤖 Selected: {selected}")
                    print("=" * 60)
                    return {"selected_model": selected, "error": ""}
                print(f"Select a number between 1 and {len(models)}.")
            except ValueError:
                print("Invalid input. Enter a number.")
    except KeyboardInterrupt:
        return {"selected_model": "", "error": "Model selection cancelled"}
    except Exception as e:
        return {"selected_model": "", "error": f"Model selection error: {e}"}


@tool
def find_python_files_tool(directory: str) -> Dict[str, Any]:
    """Find Python files in directory and subdirectories.

    Args:
        directory: Path to search.

    Returns:
        Dict: Contains 'python_files' (list of paths) and 'error'.
    """
    try:
        dir_path = Path(directory)
        files = []
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
        for py_file in dir_path.rglob("*.py"):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            if any(part.startswith(".") for part in py_file.parts):
                continue
            files.append(py_file)
        return {"python_files": [str(f) for f in sorted(files)], "error": ""}
    except Exception as e:
        return {"python_files": [], "error": f"Error finding files: {e}"}


@tool
def select_file_tool(python_files: List[str]) -> Dict[str, Any]:
    """Select a Python file from a list.

    Args:
        python_files: List of Python file paths.

    Returns:
        Dict: Contains 'selected_file' and 'error'.
    """
    try:
        if not python_files:
            return {"selected_file": "", "error": "No Python files available"}
        print("\n📁 Python files:")
        print("-" * 50)
        for i, path in enumerate(python_files, 1):
            try:
                print(f"{i:2d}. {Path(path).relative_to(Path.cwd())}")
            except ValueError:
                print(f"{i:2d}. {path}")
        print("-" * 50)
        while True:
            user_input = (
                input(f"\nSelect file (1-{len(python_files)}): ")
                .strip()
                .lower()
            )
            try:
                file_index = int(user_input) - 1
                if 0 <= file_index < len(python_files):
                    sel = python_files[file_index]
                    try:
                        rel = Path(sel).relative_to(Path.cwd())
                        print(f"\n📁 Selected file: {rel}")
                    except ValueError:
                        print(f"\n📁 Selected file: {sel}")
                    return {"selected_file": sel, "error": ""}
                print(f"❌ Invalid number. Use 1-{len(python_files)}.")
            except ValueError:
                print("❌ Enter a valid number.")
    except KeyboardInterrupt:
        return {"selected_file": "", "error": "File selection cancelled"}
    except Exception as e:
        return {"selected_file": "", "error": f"Error selecting file: {e}"}


@tool
def load_file_tool(file_path: str) -> Dict[str, Any]:
    """Load and validate a Python file.

    Args:
        file_path: Path to the Python file.

    Returns:
        Dict: Contains 'file_content' and 'error'.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "file_content": "",
                "error": f"File not found: {file_path}",
            }
        if path.suffix != ".py":
            return {"file_content": "", "error": f"Must be .py: {file_path}"}
        with open(path, "r", encoding="utf-8") as f:
            return {"file_content": f.read(), "error": ""}
    except Exception as e:
        return {"file_content": "", "error": f"Error loading file: {e}"}


@tool
def extract_docstrings_tool(code: str) -> Dict[str, Any]:
    """Extract docstrings from Python code.

    Args:
        code: Python code to analyze.

    Returns:
        Dict: Contains 'docstring_info' (list of details) and 'error'.
    """
    try:
        tree = ast.parse(code)
        docs = []
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            docs.append({
                "type": "module",
                "name": "__module__",
                "lineno": tree.body[0].lineno,
                "docstring": tree.body[0].value.value,
            })
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
            ):
                info = {
                    "type": (
                        "function"
                        if isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                        else "class"
                    ),
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": None,
                }
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    info["docstring"] = node.body[0].value.value
                docs.append(info)
        return {"docstring_info": docs, "error": ""}
    except SyntaxError as e:
        return {"docstring_info": [], "error": f"Invalid syntax: {e}"}
    except Exception as e:
        return {"docstring_info": [], "error": f"Error extracting: {e}"}


@tool
def remove_docstrings_and_comments_tool(code: str) -> Dict[str, Any]:
    """Remove all docstrings and comments from Python code.

    Args:
        code: String containing the Python code to process.

    Returns:
        Dict:
            Contains 'processed_code' (code without docstrings/comments)
            and 'error'.
    """

    def _minimal_syntax_fix(code: str) -> str:
        """Apply minimal syntax fixes to ensure valid Python code.

        Args:
            code: Code string to fix.

        Returns:
            str: Fixed code with 'pass' added to empty blocks if needed.
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

    try:
        tree = ast.parse(code)
        lines = code.split("\n")
        docstring_ranges = []
        comment_lines = set()
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
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
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
        string_io = StringIO(code)
        tokens = list(tokenize.generate_tokens(string_io.readline))
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_lines.add(token.start[0] - 1)
        all_ranges = docstring_ranges + [
            {"start": line, "end": line, "type": "comment"}
            for line in comment_lines
        ]
        all_ranges.sort(key=lambda x: x["start"], reverse=True)
        for range_info in all_ranges:
            start_line = range_info["start"]
            end_line = range_info["end"]
            if start_line < len(lines):
                indent_match = lines[start_line].lstrip()
                indent_level = len(lines[start_line]) - len(indent_match)
            else:
                indent_level = 0
            del lines[start_line : end_line + 1]
            if range_info["type"] == "module":
                while (
                    start_line < len(lines) and not lines[start_line].strip()
                ):
                    del lines[start_line]
            if range_info["type"] == "function_class" and range_info.get(
                "is_only_body", False
            ):
                lines.insert(start_line, " " * indent_level + "pass")
        result = "\n".join(lines)
        try:
            ast.parse(result)
        except SyntaxError:
            result = _minimal_syntax_fix(result)
        return {"processed_code": result, "error": ""}
    except Exception as e:
        return {
            "processed_code": "",
            "error": f"Error removing docstrings and comments: {str(e)}",
        }


@tool
def save_file_tool(
    content: str, output_dir: str, orig_path: str
) -> Dict[str, Any]:
    """Save processed code to a file.

    Args:
        content: Code to save.
        output_dir: Directory to save the file.
        orig_path: Original file path for filename.

    Returns:
        Dict: Contains 'saved_file' (path) and 'error'.
    """
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        orig = Path(orig_path)
        out_path = out_dir / orig.name
        if not content.endswith("\n"):
            content += "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"saved_file": str(out_path), "error": ""}
    except Exception as e:
        return {"saved_file": "", "error": f"Error saving file: {e}"}
