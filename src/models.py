"""
Model wrappers used by the reproduction scripts.

This repo expects `from src.models import HarmFormer`.

The paper's HarmFormer is released on HuggingFace with a non-standard `config.json`.
We load the referenced Longformer base encoder and then load the released
`pytorch_model.bin` weights (which contain `base_model.*` and `classifiers.*`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)


_HARM_ATTRS: Dict[str, str] = {
    "H": "hate_violence",
    "IH": "ideological",
    "SE": "sexual",
    "IL": "illegal",
    "SI": "self_inflicted",
}


@dataclass(frozen=True)
class _HFHarmFormerConfig:
    base_model_name: str
    num_classes: int
    num_risk_levels: int
    max_length: int
    class_map_dict: Dict[int, str]  # idx -> harm short code (e.g., 0 -> "H")


class _HarmFormerTorchModel:  # minimal nn.Module wrapper, not a PreTrainedModel
    def __init__(self, base_model, classifiers, torch):
        self.torch = torch
        self.base_model = base_model
        self.classifiers = classifiers

    def to(self, device: str):
        self.base_model.to(device)
        self.classifiers.to(device)
        return self

    def eval(self):
        self.base_model.eval()
        self.classifiers.eval()
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Prefer pooler_output when present; fallback to CLS token.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs[0][:, 0, :]

        logits_per_class = []
        for clf in self.classifiers:
            logits_per_class.append(clf(pooled))
        # (batch, num_classes, num_risk_levels)
        return self.torch.stack(logits_per_class, dim=1)


class HarmFormer:
    """
    Wrapper around `themendu/HarmFormer`.

    The upstream model returns per-harm logits for 3 risk levels.
    We interpret them as: [safe, topical, toxic].
    """

    def __init__(
        self,
        model_name: str = "themendu/HarmFormer",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
            try:
                # Import the base encoder directly to avoid pulling in transformers'
                # generation stack (which can drag in heavy optional deps).
                from transformers.models.longformer.modeling_longformer import (  # type: ignore
                    LongformerModel,
                )
            except Exception:  # pragma: no cover - fallback for older transformers
                from transformers import LongformerModel  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "HarmFormer requires: pip install torch transformers accelerate huggingface-hub"
            ) from e

        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Download + parse the repo's custom config.json
        cfg_path = hf_hub_download(repo_id=model_name, filename="config.json")
        raw_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self._cfg = _HFHarmFormerConfig(
            base_model_name=str(raw_cfg.get("model_name", "allenai/longformer-base-4096")),
            num_classes=int(raw_cfg.get("num_classes", 5)),
            num_risk_levels=int(raw_cfg.get("num_risk_levels", 3)),
            max_length=int(raw_cfg.get("max_length", 1024)),
            class_map_dict={int(k): str(v) for k, v in (raw_cfg.get("class_map_dict") or {}).items()},
        )

        logger.info(f"Loading HarmFormer from {model_name} on {self.device}...")
        # Load base encoder and build heads matching the released checkpoint.
        # Explicitly disable safetensors auto-conversion (which can trigger network calls
        # and noisy background-thread timeouts on some HPC networks).
        base_model = LongformerModel.from_pretrained(self._cfg.base_model_name, use_safetensors=False)
        hidden_size = int(getattr(base_model.config, "hidden_size"))
        classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, self._cfg.num_risk_levels),
                )
                for _ in range(self._cfg.num_classes)
            ]
        )

        model = _HarmFormerTorchModel(base_model=base_model, classifiers=classifiers, torch=self.torch)

        # Load checkpoint weights from HF Hub.
        ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = self.torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.base_model.load_state_dict(
            {k[len("base_model.") :]: v for k, v in state_dict.items() if k.startswith("base_model.")},
            strict=False,
        )
        # Load classifier weights
        clf_state = {k[len("classifiers.") :]: v for k, v in state_dict.items() if k.startswith("classifiers.")}
        classifiers.load_state_dict(clf_state, strict=True)

        if unexpected:
            logger.warning(f"Unexpected keys when loading HarmFormer base_model: {unexpected[:10]}")
        if missing:
            # Longformer variants can have small differences; warn but continue.
            logger.warning(f"Missing keys when loading HarmFormer base_model: {missing[:10]}")

        self.model = model.to(self.device).eval()

        # Use the base model tokenizer (Longformer) for consistent tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(self._cfg.base_model_name, use_fast=True)

        # Default mapping if class_map_dict is missing.
        if not self._cfg.class_map_dict:
            self._cfg = _HFHarmFormerConfig(
                base_model_name=self._cfg.base_model_name,
                num_classes=self._cfg.num_classes,
                num_risk_levels=self._cfg.num_risk_levels,
                max_length=self._cfg.max_length,
                class_map_dict={0: "H", 1: "IH", 2: "SE", 3: "IL", 4: "SI"},
            )

    def _encode_batch(self, texts: List[str]):
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self._cfg.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def get_probabilities(self, text: str) -> Dict[str, Dict[str, float]]:
        text = (text or "").strip()
        if not text:
            return {attr: {"safe": 1.0, "topical": 0.0, "toxic": 0.0} for attr in _HARM_ATTRS.values()}

        inputs = self._encode_batch([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            logits = self.model(**inputs)  # (1, num_classes, num_risk_levels)
            probs = self.torch.softmax(logits, dim=-1)[0]

        out: Dict[str, Dict[str, float]] = {}
        for class_idx in range(self._cfg.num_classes):
            harm_code = self._cfg.class_map_dict.get(class_idx)
            if harm_code not in _HARM_ATTRS:
                continue
            p = probs[class_idx].tolist()
            # risk level order in upstream code is softmax over last dim; we interpret as [safe, topical, toxic]
            out[_HARM_ATTRS[harm_code]] = {"safe": float(p[0]), "topical": float(p[1]), "toxic": float(p[2])}

        # Ensure all harms exist (fail-safe to safe).
        for harm_code, attr in _HARM_ATTRS.items():
            out.setdefault(attr, {"safe": 1.0, "topical": 0.0, "toxic": 0.0})

        return out

    def predict(self, text: str) -> HarmLabel:
        probs = self.get_probabilities(text)
        label = HarmLabel()
        for harm_code, attr in _HARM_ATTRS.items():
            p = probs[attr]
            best = max(p.items(), key=lambda kv: kv[1])[0]
            if best == "safe":
                dim = Dimension.SAFE
            elif best == "topical":
                dim = Dimension.TOPICAL
            else:
                dim = Dimension.TOXIC
            setattr(label, attr, dim)
        return label

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[HarmLabel]:
        if not texts:
            return []

        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(range(0, len(texts), self.batch_size), desc="HarmFormer", unit="batch")
            except ImportError:
                iterator = range(0, len(texts), self.batch_size)
        else:
            iterator = range(0, len(texts), self.batch_size)

        labels: List[HarmLabel] = []
        for i in iterator:
            batch = texts[i : i + self.batch_size]
            inputs = self._encode_batch(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                logits = self.model(**inputs)
                probs = self.torch.softmax(logits, dim=-1)

            for row in probs:
                label = HarmLabel()
                for class_idx in range(self._cfg.num_classes):
                    harm_code = self._cfg.class_map_dict.get(class_idx)
                    if harm_code not in _HARM_ATTRS:
                        continue
                    p = row[class_idx].tolist()
                    best_idx = int(self.torch.argmax(row[class_idx]).item())
                    dim = Dimension.SAFE if best_idx == 0 else (Dimension.TOPICAL if best_idx == 1 else Dimension.TOXIC)
                    setattr(label, _HARM_ATTRS[harm_code], dim)
                labels.append(label)

        return labels

