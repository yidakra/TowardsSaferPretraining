import os

import pytest


pytestmark = [pytest.mark.slow]


if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Integration smoke test. Set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


def test_core_components_importable():
    # Arrange / Act
    import src  # noqa: F401
    from src.data_loaders import HAVOCLoader, TTPEvalLoader  # noqa: F401
    from src.evaluation import HAVOCEvaluator, TTPEvaluator  # noqa: F401


@pytest.mark.requires_model
def test_transformers_generator_smoke():
    # Arrange
    from src.evaluation.llm_generator import create_generator

    gen = create_generator("transformers", "gpt2")

    # Act
    out = gen.generate("Hello there!", max_tokens=5, temperature=0.0)

    # Assert
    assert isinstance(out, str)
    assert len(out) > 0

