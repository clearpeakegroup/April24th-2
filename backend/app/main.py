from api import app
from api.router import router
import os
import asyncio
from backend.data_ingestion.sentiment_feature_pipeline import run_sentiment_feature_pipeline
from backend.data_ingestion.databento_ingest import DatabentoIngestor

# The app is already created and router included in api/__init__.py
# Only add the startup event if needed
if os.getenv("RUN_SENTIMENT_PIPELINE_ON_STARTUP", "0") == "1":
    @app.on_event("startup")
    async def start_sentiment_pipeline():
        asyncio.create_task(run_sentiment_feature_pipeline())

@app.on_event("startup")
async def start_databento_ingest():
    ingestor = DatabentoIngestor(
        dataset="GLBX.MDP3",  # <FILL_ME> or update as needed
        symbols=["ESZ3", "NQZ3"],  # <FILL_ME> or update as needed
        schema="mbo"
    )
    asyncio.create_task(ingestor.ingest_forever())
