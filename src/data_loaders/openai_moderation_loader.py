"""
Loader for the OpenAI moderation evaluation dataset (Markov et al., 2022).

Paper context (Mendu et al., 2025 - Table 7):
- Uses the OpenAI Moderation test set: 1680 samples = 1158 non-toxic + 522 toxic
- Toxic label is defined as any of the category flags being 1.

Dataset source (vendored via git clone):
`data/moderation-api-release/data/samples-1680.jsonl.gz`
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List


OPENAI_MODERATION_KEYS: List[str] = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


@dataclass(frozen=True)
class OpenAIModerationSample:
    text: str
    is_toxic: bool


class OpenAIModerationLoader:
    def __init__(self, dataset_path: str = "data/moderation-api-release/data/samples-1680.jsonl.gz"):
        self.dataset_path = Path(dataset_path)

    def _iter_jsonl_gz(self) -> Iterable[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Moderation dataset not found: {self.dataset_path}")
        with gzip.open(self.dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                yield json.loads(line)

    def load(self, limit: Optional[int] = None) -> List[OpenAIModerationSample]:
        samples: List[OpenAIModerationSample] = []
        for obj in self._iter_jsonl_gz():
            text = (obj.get("prompt") or "").strip()
            if not text:
                continue
            is_toxic = any(int(obj.get(k, 0) or 0) == 1 for k in OPENAI_MODERATION_KEYS)
            samples.append(OpenAIModerationSample(text=text, is_toxic=is_toxic))
            if limit is not None and len(samples) >= limit:
                break
        return samples

