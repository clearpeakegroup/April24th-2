from fastapi import FastAPI
from api import app
import os
import asyncio
from backend.data_ingestion.sentiment_feature_pipeline import run_sentiment_feature_pipeline

app = FastAPI()
app.include_router(router)

if os.getenv("RUN_SENTIMENT_PIPELINE_ON_STARTUP", "0") == "1":
    @app.on_event("startup")
    async def start_sentiment_pipeline():
        asyncio.create_task(run_sentiment_feature_pipeline())
