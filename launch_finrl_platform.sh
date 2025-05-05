#!/usr/bin/env bash
set -e

LOGDIR="logs"
mkdir -p "$LOGDIR"
HEALTH_LOG="$LOGDIR/health_check_failures.log"
echo "Health check failures (if any):" > "$HEALTH_LOG"

# Health checks (do not exit on failure, just log)
SERVICES=(
  "backend.data_ingestion.news_benzinga"
  "backend.data_ingestion.news_marketwatch"
  "backend.data_ingestion.news_twitter_x"
  "backend.data_ingestion.news_macroedge"
  "backend.data_ingestion.databento_ingest"
  "backend.data_ingestion.live_stream"
  "backend.data_ingestion.sentiment_feature_pipeline"
  "backend.data_ingestion.orderbook_feature_pipeline"
  "backend.model_core.model_runner"
  "backend.model_core.live_inference_loop"
)

for svc in "${SERVICES[@]}"; do
  python3 -c "from $svc import health_check; assert health_check()" 2>> "$HEALTH_LOG" || \
    echo "$svc health check failed (see $HEALTH_LOG)" >> "$HEALTH_LOG"
done

echo "All health checks attempted. See $HEALTH_LOG for any failures. Starting FinRL Platform..."

# Start backend services (run each in background, log output)
nohup python3 -m backend.data_ingestion.news_benzinga > "$LOGDIR/news_benzinga.log" 2>&1 &
nohup python3 -m backend.data_ingestion.news_marketwatch > "$LOGDIR/news_marketwatch.log" 2>&1 &
nohup python3 -m backend.data_ingestion.news_twitter_x > "$LOGDIR/news_twitter_x.log" 2>&1 &
nohup python3 -m backend.data_ingestion.news_macroedge > "$LOGDIR/news_macroedge.log" 2>&1 &
nohup python3 -m backend.data_ingestion.databento_ingest > "$LOGDIR/databento_ingest.log" 2>&1 &
nohup python3 -m backend.data_ingestion.live_stream > "$LOGDIR/live_stream.log" 2>&1 &
nohup python3 -m backend.data_ingestion.sentiment_feature_pipeline > "$LOGDIR/sentiment_feature_pipeline.log" 2>&1 &
nohup python3 -m backend.data_ingestion.orderbook_feature_pipeline > "$LOGDIR/orderbook_feature_pipeline.log" 2>&1 &
nohup python3 -m backend.model_core.live_inference_loop > "$LOGDIR/live_inference_loop.log" 2>&1 &

# Optionally start Gradio UI if present
if [ -f "frontend-ui/app.py" ]; then
  nohup python3 frontend-ui/app.py > "$LOGDIR/frontend_ui.log" 2>&1 &
  echo "Gradio UI started."
fi

echo "FinRL Platform launched. Logs in $LOGDIR/."
echo "To stop: killall python3 or pkill -f backend."
echo "Check $HEALTH_LOG for any health check failures." 