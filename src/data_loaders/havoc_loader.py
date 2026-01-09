"""
HAVOC dataset loader.

HAVOC (Harmful Abstractions and Violations in Open Completions) is a benchmark
for evaluating LLM safety across multiple harm dimensions.
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import ast

from ..utils.taxonomy import HarmLabel, Dimension


@dataclass
class HAVOCSample:
    """Single sample from HAVOC dataset."""

    # Core fields
    prefix: str
    suffix: str
    prefix_label: HarmLabel

    # Optional fields from model evaluation
    snippet: Optional[str] = None
    prefix_gen_gpt_response: Optional[str] = None

    # Model responses and judgments
    llama_3b_response: Optional[str] = None
    llama_3b_judge: Optional[str] = None
    llama_3b_label: Optional[str] = None

    llama_1b_response: Optional[str] = None
    llama_1b_judge: Optional[str] = None
    llama_1b_label: Optional[str] = None

    mistral_7b_response: Optional[str] = None
    mistral_7b_judge: Optional[str] = None
    mistral_7b_label: Optional[str] = None

    gemma_2b_response: Optional[str] = None
    gemma_2b_judge: Optional[str] = None
    gemma_2b_label: Optional[str] = None

    gemma_9b_response: Optional[str] = None
    gemma_9b_judge: Optional[str] = None
    gemma_9b_label: Optional[str] = None

    gemma_27b_response: Optional[str] = None
    gemma_27b_judge: Optional[str] = None
    gemma_27b_label: Optional[str] = None

    def get_full_text(self) -> str:
        """Get concatenated prefix + suffix."""
        return self.prefix + " " + self.suffix

    def is_toxic_prefix(self) -> bool:
        """Check if prefix is toxic."""
        return self.prefix_label.is_toxic()

    def is_topical_prefix(self) -> bool:
        """Check if prefix is topical."""
        return self.prefix_label.is_topical()

    def is_safe_prefix(self) -> bool:
        """Check if prefix is safe."""
        return self.prefix_label.is_safe()

    def get_leakage_type(self) -> str:
        """
        Determine leakage type based on prefix label.

        Returns:
            "provocative" if prefix is toxic
            "passive" if prefix is topical
            "neutral" if prefix is safe
        """
        if self.is_toxic_prefix():
            return "provocative"
        elif self.is_topical_prefix():
            return "passive"
        else:
            return "neutral"


class HAVOCLoader:
    """
    Loader for HAVOC dataset.

    Example usage:
        loader = HAVOCLoader("data/HAVOC/havoc.tsv")
        samples = loader.load()
        print(f"Loaded {len(samples)} samples")

        # Filter by leakage type
        provocative = loader.filter_by_leakage_type("provocative")
    """

    def __init__(self, filepath: str, modeleval_filepath: Optional[str] = None):
        """
        Initialize loader.

        Args:
            filepath: Path to havoc.tsv file
            modeleval_filepath: Optional path to havoc_modeleval.tsv
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"HAVOC file not found: {filepath}")

        self.modeleval_filepath = Path(modeleval_filepath) if modeleval_filepath else None
        self._samples: Optional[List[HAVOCSample]] = None

    def _parse_label_list(self, label_str: str) -> HarmLabel:
        """
        Parse label from string list format.

        Args:
            label_str: String like "['topical', 'none', 'none', 'none', 'none']"

        Returns:
            HarmLabel object
        """
        try:
            # Use ast.literal_eval for safe parsing
            labels = ast.literal_eval(label_str)

            # Handle empty or malformed labels
            if not labels or len(labels) != 5:
                print(f"Warning: Invalid label list '{label_str}', defaulting to all safe")
                return HarmLabel()  # Returns all safe

            return HarmLabel.from_list(labels)
        except Exception as e:
            print(f"Warning: Failed to parse label '{label_str}': {e}. Defaulting to all safe")
            return HarmLabel()  # Returns all safe

    def load(self, force_reload: bool = False) -> List[HAVOCSample]:
        """
        Load all samples from the dataset.

        Args:
            force_reload: If True, reload even if already loaded

        Returns:
            List of HAVOCSample objects
        """
        if self._samples is not None and not force_reload:
            return self._samples

        samples = []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                prefix_label = self._parse_label_list(row['PrefixLab'])

                sample = HAVOCSample(
                    prefix=row['Prefix'],
                    suffix=row['Suffix'],
                    prefix_label=prefix_label,
                )
                samples.append(sample)

        # Load model evaluations if available
        if self.modeleval_filepath and self.modeleval_filepath.exists():
            self._load_model_evaluations(samples)

        self._samples = samples
        return samples

    def _load_model_evaluations(self, samples: List[HAVOCSample]):
        """Load model evaluation data and merge with samples."""
        with open(self.modeleval_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for idx, row in enumerate(reader):
                if idx >= len(samples):
                    break

                sample = samples[idx]

                # Update with model evaluation data
                sample.snippet = row.get('Snippet')
                sample.prefix_gen_gpt_response = row.get('PrefixGenGPTResponse')

                sample.llama_3b_response = row.get('PrefixLlama3BResponse')
                sample.llama_3b_judge = row.get('Llama3BJudge')
                sample.llama_3b_label = row.get('Llama3BLab')

                sample.llama_1b_response = row.get('PrefixLlama1BResponse')
                sample.llama_1b_judge = row.get('Llama1BJudge')
                sample.llama_1b_label = row.get('Llama1BLab')

                sample.mistral_7b_response = row.get('PrefixMistral7BResponse')
                sample.mistral_7b_judge = row.get('Mistral7BJudge')
                sample.mistral_7b_label = row.get('Mistral7BLab')

                sample.gemma_2b_response = row.get('PrefixGemma2BResponse')
                sample.gemma_2b_judge = row.get('Gemma2BJudge')
                sample.gemma_2b_label = row.get('Gemma2BLab')

                sample.gemma_9b_response = row.get('PrefixGemma9BResponse')
                sample.gemma_9b_judge = row.get('Gemma9BJudge')
                sample.gemma_9b_label = row.get('Gemma9BLab')

                sample.gemma_27b_response = row.get('PrefixGemma27BResponse')
                sample.gemma_27b_judge = row.get('Gemma27BJudge')
                sample.gemma_27b_label = row.get('Gemma27BLab')

    def filter_by_leakage_type(self, leakage_type: str) -> List[HAVOCSample]:
        """
        Filter samples by leakage type.

        Args:
            leakage_type: One of "provocative", "passive", "neutral"

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        valid_types = ["provocative", "passive", "neutral"]
        if leakage_type not in valid_types:
            raise ValueError(f"Invalid leakage type. Must be one of: {valid_types}")

        return [s for s in self._samples if s.get_leakage_type() == leakage_type]

    def filter_by_harm(
        self,
        harm_category: str,
        dimension: Optional[Dimension] = None
    ) -> List[HAVOCSample]:
        """
        Filter samples by specific harm category in prefix.

        Args:
            harm_category: One of "H", "IH", "SE", "IL", "SI"
            dimension: Optional dimension filter (SAFE, TOPICAL, TOXIC)

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        harm_map = {
            "H": "hate_violence",
            "IH": "ideological",
            "SE": "sexual",
            "IL": "illegal",
            "SI": "self_inflicted",
        }

        if harm_category not in harm_map:
            raise ValueError(f"Invalid harm category: {harm_category}")

        attr_name = harm_map[harm_category]
        filtered = []

        for sample in self._samples:
            label = getattr(sample.prefix_label, attr_name)
            if dimension is None or label == dimension:
                filtered.append(sample)

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with various statistics
        """
        if self._samples is None:
            self.load()

        total = len(self._samples)

        # Leakage type distribution
        provocative = len(self.filter_by_leakage_type("provocative"))
        passive = len(self.filter_by_leakage_type("passive"))
        neutral = len(self.filter_by_leakage_type("neutral"))

        # Per-harm statistics
        harm_stats = {}
        for short_name, attr_name in [
            ("H", "hate_violence"),
            ("IH", "ideological"),
            ("SE", "sexual"),
            ("IL", "illegal"),
            ("SI", "self_inflicted"),
        ]:
            toxic = len([s for s in self._samples
                        if getattr(s.prefix_label, attr_name) == Dimension.TOXIC])
            topical = len([s for s in self._samples
                          if getattr(s.prefix_label, attr_name) == Dimension.TOPICAL])
            safe = len([s for s in self._samples
                       if getattr(s.prefix_label, attr_name) == Dimension.SAFE])
            harm_stats[short_name] = {
                "toxic": toxic,
                "topical": topical,
                "safe": safe,
            }

        return {
            "total_samples": total,
            "leakage_types": {
                "provocative": provocative,
                "passive": passive,
                "neutral": neutral,
            },
            "per_harm_statistics": harm_stats,
        }

    def __len__(self) -> int:
        """Get number of samples."""
        if self._samples is None:
            self.load()
        return len(self._samples)

    def __getitem__(self, idx: int) -> HAVOCSample:
        """Get sample by index."""
        if self._samples is None:
            self.load()
        return self._samples[idx]
