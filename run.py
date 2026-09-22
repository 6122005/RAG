import os
import sys
from pathlib import Path

# If run from repository root, add backend to sys.path
root_dir = Path(__file__).resolve().parent
backend_dir = root_dir / "backend"
if backend_dir.exists() and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn

if __name__ == "__main__":
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"--> [Root Runner] Starting Uvicorn on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port)
