import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SANDBOX_DIR = Path(os.getenv("AGENT_SANDBOX_DIR", "./agent_workspace")).resolve()

ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xlsx", ".xls", ".py", ".yaml", ".yml"}


def normalize_filepath(filepath: str) -> str:
    """
    Strips sandbox prefix from filepath if LLM accidentally includes it.

    Examples:
        'agent_workspace/sales.xlsx' → 'sales.xlsx'
        './agent_workspace/sales.xlsx' → 'sales.xlsx'
        '/abs/path/agent_workspace/sales.xlsx' → 'sales.xlsx'
        'sales.xlsx' → 'sales.xlsx'  (unchanged)
        'reports/q1.xlsx' → 'reports/q1.xlsx'  (unchanged)
    """
    path = Path(filepath)

    # Try to make it relative to SANDBOX_DIR
    try:
        return str(path.resolve().relative_to(SANDBOX_DIR))
    except ValueError:
        pass

    # Try to strip sandbox dir name as a prefix (e.g. 'agent_workspace/sales.xlsx')
    sandbox_name = SANDBOX_DIR.name
    parts = path.parts
    if parts and parts[0] in {sandbox_name, f"./{sandbox_name}"}:
        return str(Path(*parts[1:]))

    # Already a clean relative path
    return filepath


def get_safe_path(filepath: str, allowed_extensions: set = None) -> Path:
    """
    Resolves the path and ensures it stays inside the sandbox.
    Raises ValueError if the path tries to escape the sandbox.

    Args:
        filepath: relative path inside sandbox
        allowed_extensions: override default allowed extensions if needed
    """
    extensions = allowed_extensions or ALLOWED_EXTENSIONS
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    clean_filepath = normalize_filepath(filepath)
    full_path = (SANDBOX_DIR / clean_filepath).resolve()
    if not str(full_path).startswith(str(SANDBOX_DIR)):
        raise ValueError(f"Access denied: '{filepath}' is outside the workspace.")
    if full_path.suffix.lower() not in extensions:
        raise ValueError(f"File type '{full_path.suffix}' is not allowed. Allowed: {extensions}")
    return full_path