"""``python -m webapp [--host 127.0.0.1] [--port 8000]``: arranca la demo web local."""
import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="ShopAssist · demo web local (FastAPI + Ollama)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"[webapp] abre http://localhost:{a.port} (los modelos tardan en cargar; progreso en /api/health)")
    uvicorn.run("webapp.server:app", host=a.host, port=a.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
