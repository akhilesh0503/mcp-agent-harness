import asyncio
import json
import os

_BASE_DIR = os.getenv("FILE_READ_BASE_DIR", "/tmp/agent_files")
_MAX_BYTES = 1 * 1024 * 1024  # 1 MB cap


def _read_sync(abs_path: str) -> str:
    with open(abs_path, "r", encoding="utf-8") as fh:
        return fh.read(_MAX_BYTES)


def _resolve_and_validate(path: str) -> tuple[str | None, str | None]:
    """Return (abs_path, error). One of them will be None."""
    real_base = os.path.realpath(_BASE_DIR)
    abs_path = os.path.realpath(os.path.join(real_base, path))

    # Reject anything that escapes the sandbox
    if not (abs_path == real_base or abs_path.startswith(real_base + os.sep)):
        return None, "Access denied: path is outside the allowed directory"

    if not os.path.exists(abs_path):
        return None, f"File not found: {path}"

    if not os.path.isfile(abs_path):
        return None, f"Not a file: {path}"

    return abs_path, None


async def file_read_tool(path: str) -> str:
    abs_path, err = _resolve_and_validate(path)
    if err:
        return json.dumps({"error": err})

    try:
        content = await asyncio.to_thread(_read_sync, abs_path)
        return json.dumps({"path": path, "content": content})
    except PermissionError:
        return json.dumps({"error": "Permission denied"})
    except UnicodeDecodeError:
        return json.dumps({"error": "File is not valid UTF-8 text"})
    except Exception as e:
        return json.dumps({"error": str(e)})
