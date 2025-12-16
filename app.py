"""
🎤 EpigrafIA API - Render Entry Point
======================================
"""

import os
import sys
from pathlib import Path

# Añadir backend al path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Importar app desde backend.main
from main import app

# Para ejecutar localmente
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)