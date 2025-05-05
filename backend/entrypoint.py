"""
Import‑proxy so `uvicorn entrypoint:app` never breaks
even if you later move the real FastAPI instance elsewhere.
"""
# Import the app from the correct location
# from backend.app.main import app 
from backend.api import app
