from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rapidfuzz import process, fuzz


@dataclass
class PriceResult:
    product_name: str
    sku: str
    price_usd: float
    score: float
    latency_ms: float


class CSVPriceTool:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def lookup(self, query: str) -> Optional[PriceResult]:
        t0 = time.perf_counter()
        choices = self.df["product_name"].astype(str).tolist()
        match = process.extractOne(query, choices, scorer=fuzz.WRatio)
        if not match:
            return None
        name, score, idx = match
        row = self.df.iloc[idx]
        dt = (time.perf_counter() - t0) * 1000
        return PriceResult(
            product_name=row["product_name"],
            sku=row["sku"],
            price_usd=float(row["price_usd"]),
            score=float(score),
            latency_ms=dt,
        )


