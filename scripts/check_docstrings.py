"""Check a Python source file for missing docstrings and type hints.

Reads JSON from stdin (provided by the PostFileSave hook) to find the
saved file path, then uses ``ast`` to inspect every function and method.

Usage (called by the hook):
    python scripts/check_docstrings.py < file_path.json

Exit codes:
    0 — all checks passed (or non-Python file)
    1 — one or more issues found
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def _extract_file_path(payload: dict) -> str | None:
    """Try several common keys to find the saved file path in the hook JSON.

    Args:
        payload: Parsed JSON dict from stdin.

    Returns:
        File path string if found, else None.
    """
    for key in ("filePath", "path", "file", "filename", "targetFile"):
        val = payload.get(key)
        if isinstance(val, str) and val.endswith(".py"):
            return val
    # Fall back to the matcher pattern if present
    for key in ("matchedPath", "matched_file"):
        val = payload.get(key)
        if isinstance(val, str):
            return val
    return None


def _func_has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function has a docstring.

    Args:
        node: AST function definition node.

    Returns:
        True if a docstring (first statement is ast.Constant with str value) exists.
    """
    if not node.body:
        return False
    first = node.body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )


def _has_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function has a return type annotation.

    Args:
        node: AST function definition node.

    Returns:
        True if ``returns`` attribute is not None.
    """
    return node.returns is not None


def _has_param_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if all non-self / non-cls parameters have type annotations.

    Args:
        node: AST function definition node.

    Returns:
        True if every argument (except self/cls) has an annotation.
    """
    for arg in node.args.args:
        # Skip self (methods) and cls (classmethods)
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return False
    return True


def check_file(file_path: str) -> list[str]:
    """Check a single Python file for missing docstrings and type hints.

    Args:
        file_path: Path to the .py file to inspect.

    Returns:
        List of issue description strings (empty if no issues).
    """
    path = Path(file_path)
    if not path.exists() or path.suffix != ".py":
        return []

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return [f"⚠ Syntax error in {file_path} — cannot check."]

    issues: list[str] = []
    mod_name = path.name

    # Module-level docstring
    if not _func_has_docstring(ast.FunctionDef(name="<module>", body=tree.body, args=ast.arguments())):
        # Check if the first statement is a docstring at module level
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
            pass  # module docstring exists
        else:
            issues.append(f"📄 {mod_name}: missing module-level docstring")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip dunder methods except __init__
            if node.name.startswith("__") and node.name != "__init__":
                continue
            qual = f"{mod_name}::{node.name}()"

            if not _func_has_docstring(node):
                issues.append(f"📝 {qual}: missing docstring")

            if not _has_param_annotations(node):
                issues.append(f"🏷 {qual}: missing parameter type hint(s)")

            if not _has_return_annotation(node):
                # __init__ and property setters don't need return annotations
                if node.name != "__init__" and not any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in getattr(node, "decorator_list", [])
                ):
                    issues.append(f"↩ {qual}: missing return type hint")

    return issues


def main() -> None:
    """Read JSON from stdin, extract the file path, and run checks."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, OSError):
        payload = {}

    file_path = _extract_file_path(payload)

    if not file_path:
        # If no path in JSON, try the first command-line argument
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            print("⚠ Could not determine saved file path from hook payload.")
            return

    issues = check_file(file_path)

    if issues:
        print(f"\n🔍 Style check for {Path(file_path).name}:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n  {len(issues)} issue(s) found. Fix before committing.")
    else:
        print(f"✅ {Path(file_path).name}: all docstrings and type hints present.")


if __name__ == "__main__":
    main()
