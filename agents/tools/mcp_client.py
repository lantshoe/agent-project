import os
from pathlib import Path
from dotenv import load_dotenv
import sys
load_dotenv()


def get_sandbox_dir() -> str:
    path = Path(os.getenv("AGENT_SANDBOX_DIR", "./agent_workspace")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

SANDBOX_DIR = get_sandbox_dir()

def get_mcp_config() -> dict:
    """
    Central config for all MCP servers.
    To add a new MCP server, just add an entry here.
    """
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    excel_path = project_root / "mcp_servers" / "excel_server.py"
    return {
        "filesystem": {
            "command": "npx",
            "args": [
                "@modelcontextprotocol/server-filesystem",
                SANDBOX_DIR
            ],
            "transport": "stdio"
        },
        "excel": {
            "command": sys.executable,
            "args": [str(excel_path)],
            "transport": "stdio",
        }
    }
