"""Start one production API worker on Render's assigned port."""

import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    port = int(os.environ.get("PORT", "8001"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "main:app",
                             "--host", "0.0.0.0", "--port", str(port), "--workers", "1"])
