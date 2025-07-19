import ast
from pathlib import Path
from typing import List


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


def extract_docstrings(code: str) -> List[dict]:
    """Extract all docstrings from Python code.

    Args:
        code: String containing the Python code to analyze.

    Returns:
        List[dict]: List of dictionaries with docstring info.

    Raises:
        ValueError: If the code contains invalid Python syntax.
    """

    def visit_function_or_class(node):
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

        return docstrings
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
