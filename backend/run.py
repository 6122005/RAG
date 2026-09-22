import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import uvicorn

if __name__ == "__main__":
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"--> [Render Runner] Starting Uvicorn on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port)
