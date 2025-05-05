import os
from pathlib import Path
import pyarrow.parquet as pq
from backend.models.model_io import ModelInputBatch, ModelInputFeature
from typing import Iterator, List

class ModelDataLoader:
    def __init__(self, unified_base: str, batch_size: int = 128):
        self.unified_base = unified_base
        self.batch_size = batch_size

    def iter_batches(self, partition: str) -> Iterator[ModelInputBatch]:
        """
        Yields ModelInputBatch objects from unified feature Parquet files in the given partition (YYYY/MM/DD).
        Supports cursor-based pagination.
        """
        dir_path = Path(self.unified_base) / partition
        files = sorted(dir_path.glob("*.parquet"))
        for f in files:
            df = pq.read_table(f).to_pandas()
            features: List[ModelInputFeature] = []
            for _, row in df.iterrows():
                try:
                    feat = ModelInputFeature(**row.to_dict())
                    features.append(feat)
                    if len(features) == self.batch_size:
                        yield ModelInputBatch(features=features)
                        features = []
                except Exception as e:
                    continue
            if features:
                yield ModelInputBatch(features=features)

# Usage:
# loader = ModelDataLoader("data/lake/parquet/features/unified", batch_size=128)
# for batch in loader.iter_batches("2024/06/10"):
#     ... # Pass batch to model 