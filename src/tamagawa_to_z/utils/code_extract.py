"""
Utility to extract code snippets (max 4 KiB each) from source files
so that LLM prompt length stays reasonable.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Dict, List


def extract_functions(path: Path, targets: List[str], max_bytes: int = 4096) -> Dict[str, str]:
    """
    Extract specific functions or classes from a Python source file.
    
    Parameters
    ----------
    path : Path
        Source file (.py)
    targets : List[str]
        Function or class names to extract
    max_bytes : int
        Truncate to this size (per snippet)

    Returns
    -------
    Dict[str, str]
        {target_name: source_code_snippet}
    """
    if not path.exists() or not path.suffix == '.py':
        return {}
    
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}
    
    out: Dict[str, str] = {}
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in targets:
            try:
                snippet = ast.get_source_segment(source, node)
                if snippet:
                    # Remove extra indentation
                    snippet = textwrap.dedent(snippet)
                    # Truncate if too long
                    if len(snippet.encode('utf-8')) > max_bytes:
                        snippet = snippet[:max_bytes] + "\n# ... (truncated)"
                    out[node.name] = snippet
            except (TypeError, ValueError):
                # ast.get_source_segment can fail in some cases
                continue
    
    return out


def extract_main_functions(path: Path, max_bytes: int = 4096) -> Dict[str, str]:
    """
    Extract main functions from a Python file (those likely to be modified).
    
    Parameters
    ----------
    path : Path
        Source file (.py)
    max_bytes : int
        Truncate to this size (per snippet)

    Returns
    -------
    Dict[str, str]
        {function_name: source_code_snippet}
    """
    if not path.exists() or not path.suffix == '.py':
        return {}
    
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}
    
    # Target function patterns that are likely to be modified
    target_patterns = [
        'harmonize', 'process', 'filter', 'score', 'calculate', 'analyze',
        'extract', 'generate', 'build', 'create', 'run', 'main'
    ]
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function name contains any target pattern
            if any(pattern in node.name.lower() for pattern in target_patterns):
                functions.append(node.name)
    
    return extract_functions(path, functions, max_bytes)


def auto_extract_from_module(module_path: Path, max_files: int = 10) -> Dict[str, Dict[str, str]]:
    """
    Automatically extract code snippets from a module directory.
    
    Parameters
    ----------
    module_path : Path
        Path to the module directory
    max_files : int
        Maximum number of files to process

    Returns
    -------
    Dict[str, Dict[str, str]]
        {file_path: {function_name: source_code}}
    """
    if not module_path.is_dir():
        return {}
    
    python_files = list(module_path.rglob("*.py"))[:max_files]
    
    result = {}
    for py_file in python_files:
        # Skip __init__.py and test files
        if py_file.name in ('__init__.py', ) or py_file.name.startswith('test_'):
            continue
            
        functions = extract_main_functions(py_file)
        if functions:
            # Use relative path as key
            rel_path = str(py_file.relative_to(module_path))
            result[rel_path] = functions
    
    return result