import ast
import tokenize
from io import StringIO
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool


@tool
def find_python_files_tool(directory: str) -> Dict[str, Any]:
    """Find all Python files in directory and subdirectories.

    Args:
        directory: Path string for the directory to search.

    Returns:
        Dict: Contains (list of file paths as strings) and 'error'.
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


@tool
def extract_docstrings_tool(code: str) -> Dict[str, Any]:
    """Extract all docstrings from Python code.

    Args:
        code: String containing the Python code to analyze.

    Returns:
        Dict: Contains (list of docstring dictionaries) and 'error'.

    Raises:
        ValueError: If the code contains invalid Python syntax.
    """

    def visit_function_or_class(node):
        """Helper to extract docstring info from function or class nodes."""
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
                docstring_info = visit_function_or_class(node)
                if docstring_info["docstring"]:
                    docstrings.append(docstring_info)

        return {"docstring_info": docstrings, "error": ""}
    except SyntaxError as e:
        return {
            "docstring_info": [],
            "error": f"Invalid Python syntax: {str(e)}",
        }
    except Exception as e:
        return {
            "docstring_info": [],
            "error": f"Error extracting docstrings: {str(e)}",
        }


@tool
def load_file_tool(file_path: str) -> Dict[str, Any]:
    """Load and validate the Python file.

    Args:
        file_path: Path string for the file to load.

    Returns:
        Dict: Contains 'file_content' (loaded code) and 'error'.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "file_content": "",
                "error": f"File not found: {file_path}",
            }

        if not path.suffix == ".py":
            return {
                "file_content": "",
                "error": f"File must be a Python file (.py): {file_path}",
            }

        with open(path, "r", encoding="utf-8") as f:
            original_code = f.read()

        return {"file_content": original_code, "error": ""}
    except Exception as e:
        return {"file_content": "", "error": f"Error loading file: {str(e)}"}


@tool
def remove_docstrings_and_comments_tool(code: str) -> Dict[str, Any]:
    """Remove all docstrings and comments from Python code.

    Args:
        code: String containing the Python code to process.

    Returns:
        Dict: Contains (code without docstrings/comments) and 'error'.
    """

    def _minimal_syntax_fix(code: str) -> str:
        """Apply minimal syntax fixes without reformatting code."""
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

            # Remove the docstring or comment lines
            del lines[start_line : end_line + 1]

            # For module docstring, remove trailing newline if present
            if range_info["type"] == "module":
                while (
                    start_line < len(lines) and not lines[start_line].strip()
                ):
                    del lines[start_line]

            # Add 'pass' for empty function/class bodies
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
