import os
import sys
from pathlib import Path

# Enforce strict single-threading for BLAS/OpenMP/PyTorch to stay under 512MB RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

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
