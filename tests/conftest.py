"""Pytest configuration and shared fixtures."""

import os
import pytest  # type: ignore
from pathlib import Path


@pytest.fixture(scope="session")
def data_dir():
    """Data directory path."""
    return (Path(__file__).parent.parent / "data").resolve()


@pytest.fixture(scope="session")
def ttp_eval_path(data_dir):
    """TTP-Eval dataset path."""
    return data_dir / "TTP-Eval" / "TTPEval.tsv"


@pytest.fixture(scope="session")
def havoc_path(data_dir):
    """HAVOC dataset path."""
    return data_dir / "HAVOC" / "havoc.tsv"


@pytest.fixture(scope="session")
def has_ttp_eval(ttp_eval_path):
    """Check if TTP-Eval dataset exists."""
    return ttp_eval_path.exists()


@pytest.fixture(scope="session")
def has_havoc(havoc_path):
    """Check if HAVOC dataset exists."""
    return havoc_path.exists()


@pytest.fixture(scope="session")
def harmformer_availability():
    """Check if HarmFormer can be instantiated and cache the instance."""
    try:
        from src.models import HarmFormer
        return HarmFormer()
    except Exception as e:
        return e


@pytest.fixture(autouse=True)
def require_model(request):
    """Skip tests marked with requires_model if the model cannot be loaded.

    The HarmFormer instance is resolved lazily via ``getfixturevalue`` and ONLY
    for tests that actually carry the marker. Taking ``harmformer_availability``
    as a direct parameter here would make this autouse fixture build HarmFormer
    on the very first test collected — downloading ~2GB of weights before any
    test runs and hanging the entire suite on a slow/offline network.
    """
    if request.node.get_closest_marker("requires_model"):
        harmformer_availability = request.getfixturevalue("harmformer_availability")
        if isinstance(harmformer_availability, Exception):
            pytest.skip(f"Model not available: {harmformer_availability}")
        # If harmformer_availability is a HarmFormer instance, the model is available, so continue


@pytest.fixture(autouse=True)
def require_api_key(request):
    """Skip tests marked with requires_api_key if API key is not set."""
    if request.node.get_closest_marker("requires_api_key"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture(autouse=True)
def require_dataset(request, has_ttp_eval, has_havoc):
    """Skip tests marked with requires_dataset, requires_ttp_eval, or requires_havoc if dataset files are missing."""
    missing_datasets = set()

    # Check for granular markers
    if request.node.get_closest_marker("requires_ttp_eval"):
        if not has_ttp_eval:
            missing_datasets.add("TTP-Eval")
    if request.node.get_closest_marker("requires_havoc"):
        if not has_havoc:
            missing_datasets.add("HAVOC")

    # Check for backward compatibility with generic requires_dataset marker
    if request.node.get_closest_marker("requires_dataset"):
        if not has_ttp_eval:
            missing_datasets.add("TTP-Eval")
        if not has_havoc:
            missing_datasets.add("HAVOC")

    if missing_datasets:
        pytest.skip(f"Required datasets not available: {', '.join(sorted(missing_datasets))}")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_dataset: mark test as requiring dataset files"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model download"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )
    config.addinivalue_line(
        "markers", "requires_ttp_eval: mark test as requiring TTP-Eval dataset"
    )
    config.addinivalue_line(
        "markers", "requires_havoc: mark test as requiring HAVOC dataset"
    )