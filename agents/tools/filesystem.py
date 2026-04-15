import os
from pathlib import Path
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
SANDBOX_DIR = Path(os.getenv("AGENT_SANDBOX_DIR", "./agent_workspace")).resolve()
MAX_FILE_SIZE_MB = int(os.getenv("AGENT_MAX_FILE_SIZE_MB", "5"))
ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xlsx", ".py", ".yaml", ".yml"}

def _get_safe_path(file_path: str) -> Path:
    """
    Resolves the path and ensures it stays inside the sandbox.
    Raises ValueError if the path tries to escape the sandbox.
    """
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    full_path = (SANDBOX_DIR / file_path).resolve()

    if not str(full_path).startswith(str(SANDBOX_DIR)):
        raise ValueError(f"Access denied: '{file_path}' is outside the workspace.")

    if full_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type '{full_path.suffix}' is not allowed.")

    return full_path


@tool
def read_file(full_path: str) -> str:
    """
    read and return the contents of a file
    """
    try:
        safe_path = _get_safe_path(full_path)
        if not safe_path.exists():
            return f"Error: File '{full_path}' not found in workspace."

        size_mb = safe_path.stat().st_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            return f"Error: File is too large ({size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."
        return safe_path.read_text(encoding="utf-8")

    except ValueError as e:
        return f"Security error: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(full_path: str, content: str) -> str:
    """
    Writes content to a file. Creates the file if it doesn't exist.
    Use this when you need to save information to a file.
    Example input filepath: "data/report.txt", content: "Hello World"
    """
    try:
        safe_path = _get_safe_path(full_path)
        size_mb = len(content.encode("utf-8")) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return f"Error: Content is too large ({size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."

        # Create subdirectories inside sandbox if needed
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

        return f"Successfully wrote to '{full_path}' in workspace."

    except ValueError as e:
        return f"Security error: {str(e)}"
    except Exception as e:
        return f"Error writing file: {str(e)}"



@tool
def list_files(subdirectory: str = ".") -> str:
    """
    Lists all files in the agent workspace or a subdirectory within it.
    Use this when you need to see what files are available.
    Example input: "." or "reports/"
    """
    try:
        safe_path = _get_safe_path(subdirectory) if subdirectory != "." else SANDBOX_DIR
        SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

        if not safe_path.exists():
            return f"Directory '{subdirectory}' not found in workspace."

        files = list(safe_path.iterdir())
        if not files:
            return f"Workspace is empty."

        output = []
        for f in sorted(files):
            size = f.stat().st_size
            kind = "DIR" if f.is_dir() else "FILE"
            output.append(f"[{kind}] {f.name} ({size} bytes)")

        return "\n".join(output)

    except ValueError as e:
        return f"Security error: {str(e)}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"
