"""Tests for data loaders."""

import os
import random
import pytest  # type: ignore
from pathlib import Path
from src.data_loaders import TTPEvalLoader, HAVOCLoader
from src.utils.taxonomy import HarmLabel

# TTP-Eval dataset constants - expected sample count range for v1.0 dataset
TTP_EVAL_MIN_COUNT = 380  # Minimum expected samples (allowing for minor dataset updates)
TTP_EVAL_MAX_COUNT = 400  # Maximum expected samples


class TestTTPEvalLoader:
    """Tests for TTP-Eval dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create TTP-Eval loader fixture."""
        # Check environment variable first, then fall back to default path
        ttp_path_str = os.environ.get("TTP_EVAL_PATH", "data/TTP-Eval/TTPEval.tsv")
        ttp_path = Path(ttp_path_str)
        if not ttp_path.exists():
            pytest.skip(f"TTP-Eval dataset not found at {ttp_path}")
        return TTPEvalLoader(str(ttp_path))

    def test_load_samples(self, loader):
        """Test loading samples."""
        samples = loader.load()
        assert len(samples) > 0
        # Verify we loaded the expected TTP-Eval v1.0 dataset (originally 393 samples)
        assert TTP_EVAL_MIN_COUNT <= len(samples) <= TTP_EVAL_MAX_COUNT

    def test_load_empty_dataset(self, tmp_path):
        """Test loading empty or malformed dataset."""
        # Create an empty file
        empty_file = tmp_path / "empty.tsv"
        empty_file.write_text("")

        # Initialize loader with empty file - should not raise in __init__
        loader = TTPEvalLoader(str(empty_file))

        # Load should return empty list for empty file
        samples = loader.load()
        assert samples == []

        # Test malformed file with missing required columns
        malformed_file = tmp_path / "malformed.tsv"
        malformed_file.write_text("URL\tLang\tBody\tHate&V Lab\tIdeologi Lab\nhttp://example.com\ten\tcontent\ttopical\ttopical\n")

        loader_malformed = TTPEvalLoader(str(malformed_file))

        # Load should raise ValueError for malformed data with missing required columns
        with pytest.raises(ValueError):
            loader_malformed.load()

    def test_sample_structure(self, loader):
        """Test sample data structure."""
        samples = loader.load()
        sample = samples[0]

        assert sample.url
        assert sample.body
        assert hasattr(sample, 'get_harm_label')

        label = sample.get_harm_label()
        assert isinstance(label, HarmLabel)

    def test_filter_toxic(self, loader):
        """Test filtering toxic samples."""
        toxic_samples = loader.filter_by_toxicity(toxic=True)
        assert len(toxic_samples) > 0
        assert all(s.is_toxic() for s in toxic_samples)

    def test_filter_safe(self, loader):
        """Test filtering safe samples."""
        safe_samples = loader.filter_by_toxicity(safe=True)
        assert len(safe_samples) > 0
        assert all(s.is_safe() for s in safe_samples)

    def test_filter_non_toxic(self, loader):
        """Test filtering non-toxic samples."""
        non_toxic_samples = loader.filter_by_toxicity(toxic=False)
        assert len(non_toxic_samples) > 0
        assert all(not s.is_toxic() for s in non_toxic_samples)

    def test_filter_non_safe(self, loader):
        """Test filtering non-safe samples."""
        non_safe_samples = loader.filter_by_toxicity(safe=False)
        assert len(non_safe_samples) > 0
        assert all(not s.is_safe() for s in non_safe_samples)

    def test_filter_both_flags(self, loader):
        """Test filtering with both toxic and safe flags set to True."""
        # Samples cannot be both toxic and safe, so this should return empty list
        both_flags_samples = loader.filter_by_toxicity(toxic=True, safe=True)
        assert len(both_flags_samples) == 0

    def test_filter_default(self, loader):
        """Test default filter behavior (no parameters)."""
        all_samples = loader.load()
        default_filtered = loader.filter_by_toxicity()
        # Default behavior should return all samples (no filtering)
        assert len(default_filtered) == len(all_samples)

    def test_statistics(self, loader):
        """Test statistics generation."""
        stats = loader.get_statistics()

        assert 'total_samples' in stats
        assert 'toxic_samples' in stats
        assert 'safe_samples' in stats

        # Verify total matches expected TTP-Eval v1.0 dataset size
        assert TTP_EVAL_MIN_COUNT <= stats['total_samples'] <= TTP_EVAL_MAX_COUNT
        # Note: toxic_samples + safe_samples may be less than total_samples
        # because some samples may be topical (harmful but not necessarily toxic)
        assert stats['toxic_samples'] + stats['safe_samples'] <= stats['total_samples']

    def test_filter_by_lang(self, loader):
        """Test filtering by language codes."""
        samples = loader.load()
        if not samples:
            pytest.skip("No samples loaded from dataset")

        # Pick the first sample's language as a target filter
        lang = (samples[0].lang or "").strip()
        if not lang:
            pytest.skip("No language code available in sample")

        filtered = loader.filter_by_lang([lang])
        assert len(filtered) > 0
        assert all((s.lang or "").strip().lower() == lang.lower() for s in filtered)


class TestHAVOCLoader:
    """Tests for HAVOC dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create HAVOC loader fixture."""
        havoc_path_str = os.getenv("HAVOC_PATH", "data/HAVOC/havoc.tsv")
        havoc_path = Path(havoc_path_str)
        if not havoc_path.exists():
            pytest.skip(f"HAVOC dataset not found at {havoc_path}")
        return HAVOCLoader(str(havoc_path))

    def test_load_samples(self, loader):
        """Test loading samples."""
        samples = loader.load()
        assert len(samples) > 0
        assert len(samples) >= 10000  # Should have ~10k samples

    def test_sample_structure(self, loader):
        """Test sample data structure."""
        samples = loader.load()
        sample = samples[0]

        assert hasattr(sample, 'prefix')
        assert hasattr(sample, 'suffix')
        assert hasattr(sample, 'prefix_label')
        assert isinstance(sample.prefix_label, HarmLabel)

    def test_leakage_type_detection(self, loader):
        """Test leakage type detection."""
        samples = loader.load()

        # Handle empty loads
        if not samples:
            pytest.skip("No samples loaded from dataset")

        # Sample up to 100 random samples for comprehensive testing
        num_samples_to_test = min(100, len(samples))
        random.seed(42)  # Make sampling deterministic for reproducible tests
        sampled_indices = random.sample(range(len(samples)), num_samples_to_test)

        for idx in sampled_indices:
            sample = samples[idx]
            leakage_type = sample.get_leakage_type()
            assert leakage_type in ["neutral", "passive", "provocative"]

    def test_leakage_type_mapping(self, loader):
        """Test leakage type mapping logic."""
        samples = loader.load()

        # Find samples of each type
        neutral = [s for s in samples if s.is_safe_prefix()]
        passive = [s for s in samples if s.is_topical_prefix()]
        provocative = [s for s in samples if s.is_toxic_prefix()]

        # Assert at least one sample of any leakage type exists
        assert len(neutral) > 0 or len(passive) > 0 or len(provocative) > 0, \
            "No samples found with any leakage type (neutral, passive, or provocative)"

        # Check mutual exclusivity on a reasonable subset (first 100 samples)
        subset = samples[:100]
        for i, sample in enumerate(subset):
            safe = sample.is_safe_prefix()
            topical = sample.is_topical_prefix()
            toxic = sample.is_toxic_prefix()
            exclusivity_sum = sum([safe, topical, toxic])
            assert exclusivity_sum == 1, \
                f"Sample {i} has non-exclusive leakage types: safe={safe}, topical={topical}, toxic={toxic} (sum={exclusivity_sum})"

        # Verify type mapping for existing samples
        if neutral:
            for sample in neutral:
                assert sample.get_leakage_type() == "neutral"
        if passive:
            for sample in passive:
                assert sample.get_leakage_type() == "passive"
        if provocative:
            for sample in provocative:
                assert sample.get_leakage_type() == "provocative"
