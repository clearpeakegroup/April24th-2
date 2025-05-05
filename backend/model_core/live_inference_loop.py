import asyncio
from datetime import datetime
from backend.data_ingestion.model_data_loader import ModelDataLoader
from backend.model_core.model_runner import ModelRunner
from backend.models.model_io import ModelOutputBatch
import os
from loguru import logger
import traceback

UNIFIED_BASE = os.getenv("UNIFIED_FEATURE_BASE", "data/lake/parquet/features/unified")
BATCH_SIZE = int(os.getenv("LIVE_BATCH_SIZE", 128))
PARTITION_FMT = "%Y/%m/%d"

def validate_config():
    missing = []
    if not UNIFIED_BASE:
        missing.append("UNIFIED_FEATURE_BASE")
    if missing:
        logger.error(f"Missing required live inference config: {', '.join(missing)}")
        raise RuntimeError(f"Missing required live inference config: {', '.join(missing)}")

def health_check():
    try:
        validate_config()
        runner = ModelRunner()
        if not runner.health_check():
            raise RuntimeError("ModelRunner health check failed")
        logger.info("Live inference loop health check passed.")
        return True
    except Exception as e:
        logger.error(f"Live inference loop health check failed: {e}")
        return False

async def live_inference_loop():
    validate_config()
    loader = ModelDataLoader(UNIFIED_BASE, batch_size=BATCH_SIZE)
    runner = ModelRunner()
    while True:
        now = datetime.utcnow()
        partition = now.strftime(PARTITION_FMT)
        try:
            for batch in loader.iter_batches(partition):
                try:
                    output: ModelOutputBatch = runner.run_inference(batch)
                except Exception as infer_err:
                    logger.error(f"Model inference error: {infer_err}\n{traceback.format_exc()}")
                    continue
                # --- Integrate with trading execution engine here ---
                for action in output.actions:
                    try:
                        logger.info(f"Live action: {action.asset} {action.action} conf={action.confidence:.2f} @ {action.timestamp}")
                        # send_action_to_execution_engine(action)
                    except Exception as action_err:
                        logger.error(f"Action dispatch error: {action_err}\n{traceback.format_exc()}")
        except Exception as batch_err:
            logger.error(f"Batch loading error: {batch_err}\n{traceback.format_exc()}")
        await asyncio.sleep(1)

# Usage:
# asyncio.run(live_inference_loop())
# Integrate send_action_to_execution_engine(action) as needed. 