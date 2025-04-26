import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class TaxLot:
    def __init__(self, qty, price, ts):
        self.qty = qty
        self.price = price
        self.ts = ts

class Treasury:
    def __init__(self, var_cap: float, outdir: str = "risk/pnl_csv"):
        self.lots: List[TaxLot] = []
        self.realised_pnl = 0.0
        self.unrealised_pnl = 0.0
        self.var_cap = var_cap
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.last_price = None
        self.trading_halted = False
    def on_trade(self, side: str, qty: float, price: float, ts: Any):
        if self.trading_halted:
            return
        if side == "buy":
            self.lots.append(TaxLot(qty, price, ts))
        elif side == "sell":
            remain = qty
            while remain > 0 and self.lots:
                lot = self.lots[0]
                take = min(lot.qty, remain)
                pnl = (price - lot.price) * take
                self.realised_pnl += pnl
                lot.qty -= take
                remain -= take
                if lot.qty == 0:
                    self.lots.pop(0)
        self._check_halt()
    def mark_to_market(self, price: float):
        self.last_price = price
        self.unrealised_pnl = sum((price - lot.price) * lot.qty for lot in self.lots)
        self._check_halt()
    def _check_halt(self):
        if self.realised_pnl <= -0.6 * self.var_cap:
            self.trading_halted = True
    def dump_csv(self):
        now = datetime.utcnow()
        fname = self.outdir / f"pnl_{now.strftime('%Y%m%d')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "realised_pnl", "unrealised_pnl", "trading_halted"])
            writer.writerow([now.isoformat(), self.realised_pnl, self.unrealised_pnl, self.trading_halted])
    def status(self) -> Dict[str, Any]:
        return {
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl,
            "trading_halted": self.trading_halted,
        } 