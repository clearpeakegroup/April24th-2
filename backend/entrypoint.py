"""
Import‑proxy so `uvicorn entrypoint:app` never breaks
even if you later move the real FastAPI instance elsewhere.
"""
from backend.app.main import app
