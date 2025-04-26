import pandas as pd
import torch
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage
from transformers import AutoTokenizer, AutoModel
import numpy as np

class LiquidODEGate(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # Simple ODE-inspired gating: tanh + residual
        h = torch.tanh(self.fc1(x))
        out = self.fc2(h) + 0.1 * torch.sum(x, dim=-1, keepdim=True)
        return out.squeeze(-1)

class StrategyHead(BaseStrategy):
    def __init__(self):
        self.model_path = "models/news_bert_minilm_quant.pt"
        self.bert_model = None
        self.tokenizer = None
        self.gate = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    def _load_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        base = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        base.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.bert_model = base.to(self.device).eval()
        self.gate = LiquidODEGate(input_dim=384).to(self.device)
    def _embed(self, texts):
        with torch.no_grad():
            tokens = self.tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt").to(self.device)
            emb = self.bert_model(**tokens).last_hidden_state.mean(dim=1)
            return emb
    def _gate_signal(self, emb):
        with torch.no_grad():
            return self.gate(emb).cpu().numpy()
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        signals = self._gate_signal(self._embed(data["headline"] + " " + data["body"]))
        trades = []
        pos = 0
        cash = 100000
        for i, row in data.iterrows():
            if signals[i] > 0.5 and pos == 0:
                price = apply_slippage(row["MES"])
                trades.append({"side": "buy", "qty": 1, "price": price})
                pos = 1
                cash -= price
            elif signals[i] < -0.5 and pos == 1:
                price = apply_slippage(row["MES"])
                trades.append({"side": "sell", "qty": 1, "price": price})
                pos = 0
                cash += price
        pnl = calc_pnl(trades, data["MES"])
        equity_curve = [100000 + pnl] * len(data)
        return BacktestResult(pnl, trades, equity_curve)
    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")
    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            text = row.get("headline", "") + " " + row.get("body", "")
            emb = self._embed([text])
            signal = self._gate_signal(emb)[0]
            if signal > 0.5:
                yield Order(symbol="MES", qty=1, price=row["MES"], side="buy")
            elif signal < -0.5:
                yield Order(symbol="MES", qty=1, price=row["MES"], side="sell") 