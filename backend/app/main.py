from fastapi import FastAPI
from api import app

app = FastAPI()
app.include_router(router)
