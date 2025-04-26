from .celery_app import celery_app
from backend.agents.registry import AGENT_REGISTRY
import random
import time
from backend.data_ingestion.sentiment_feature_pipeline import run_sentiment_feature_pipeline
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pyarrow.parquet as pq
import glob
import os
import boto3
from github import Github
from datetime import datetime
from loguru import logger

load_dotenv()
BERT_MODEL = os.getenv("BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FEATURE_BASE = os.getenv("SENTIMENT_FEATURE_BASE", "data/lake/parquet/features/sentiment")
BERT_OUT_PATH = os.getenv("BERT_OUT_PATH", "models/news_bert_minilm_quant.pt")
S3_BUCKET = os.getenv("S3_BUCKET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

def get_agent_from_params(params):
    agent_type = params.get("agent_type", "ppo")
    agent_config = params.get("agent_config", {})
    env_config = params.get("env_config", {})
    agent_cls = AGENT_REGISTRY[agent_type]
    return agent_cls(env_config, **agent_config)

@celery_app.task(name="backtest.run")
def run_backtest(params):
    agent = get_agent_from_params(params)
    # Simulate slippage, latency, market impact
    time.sleep(random.uniform(0.5, 2.0))  # Simulate latency
    slippage = random.uniform(-0.05, 0.05)  # Simulate slippage
    impact = random.uniform(-0.1, 0.1)      # Simulate market impact
    # result = agent.backtest(slippage=slippage, impact=impact)  # Plug in real logic
    return {"status": "completed", "params": params, "slippage": slippage, "impact": impact}

@celery_app.task(name="forwardtest.run")
def run_forwardtest(params):
    agent = get_agent_from_params(params)
    time.sleep(random.uniform(0.5, 2.0))  # Simulate latency
    slippage = random.uniform(-0.05, 0.05)
    impact = random.uniform(-0.1, 0.1)
    # result = agent.forwardtest(slippage=slippage, impact=impact)  # Plug in real logic
    return {"status": "completed", "params": params, "slippage": slippage, "impact": impact}

@celery_app.task(name="retrain.model")
def retrain_model(params):
    agent = get_agent_from_params(params)
    agent.train(**params.get("train_args", {}))
    agent.save(params.get("model_path", f"models/{params.get('agent_type', 'ppo')}_latest.pt"))
    return {"status": "completed", "params": params}

@celery_app.task(name="liveexec.run")
def run_live_execution(params):
    agent = get_agent_from_params(params)
    agent.load(params.get("model_path", f"models/{params.get('agent_type', 'ppo')}_latest.pt"))
    # Implement live execution logic using agent.act()
    # result = agent.live_execute()  # Implement as needed
    return {"status": "completed", "params": params}

@celery_app.task(name="features.sentiment_pipeline")
def sentiment_feature_pipeline_task():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_sentiment_feature_pipeline())

@celery_app.task(name="train_news_bert")
def train_news_bert_task(params=None):
    # Load latest parquet features
    parquet_files = sorted(glob.glob(f"{FEATURE_BASE}/**/*.parquet", recursive=True), reverse=True)
    if not parquet_files:
        return {"status": "error", "msg": "No sentiment parquet found"}
    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()
    texts = df.get("headline", "") + " " + df.get("body", "")
    labels = df.get("sentiment", 0)
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels.iloc[idx])
            return item
        def __len__(self):
            return len(self.labels)
    dataset = NewsDataset(encodings, labels)
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=1)
    # Training
    args = TrainingArguments(output_dir="/tmp/bert_out", per_device_train_batch_size=32, num_train_epochs=1, logging_steps=10, save_steps=1000, fp16=True)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    # Quantize
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(model.state_dict(), BERT_OUT_PATH)
    return {"status": "completed", "model_path": BERT_OUT_PATH, "num_samples": len(df)}

@celery_app.task(name="tag_model_nightly")
def tag_model_nightly():
    tag = datetime.utcnow().strftime("%Y%m%d_nightly")
    s3_key = f"models/{tag}.pt"
    # Upload to S3
    try:
        s3 = boto3.client("s3")
        s3.upload_file(BERT_OUT_PATH, S3_BUCKET, s3_key)
        logger.info(f"Uploaded model to S3: {S3_BUCKET}/{s3_key}")
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return {"status": "error", "msg": str(e)}
    # GitHub release
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        release = repo.create_git_release(tag, tag, f"Nightly model {tag}")
        with open(BERT_OUT_PATH, "rb") as f:
            release.upload_asset(BERT_OUT_PATH, name=f"{tag}.pt")
        logger.info(f"Created GitHub release: {tag}")
    except Exception as e:
        logger.error(f"GitHub release failed: {e}")
        return {"status": "error", "msg": str(e)}
    return {"status": "completed", "tag": tag} 