import os

import pytest


pytestmark = [pytest.mark.requires_model, pytest.mark.slow]


if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Integration test (downloads model). Set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


def test_harmformer_predict_and_probabilities_smoke():
    # Arrange
    from src.models import HarmFormer

    model = HarmFormer()
    text = "This is a neutral text about cooking and gardening."

    # Act
    label = model.predict(text)
    probs = model.get_probabilities(text)

    # Assert
    assert hasattr(label, "to_dict")
    as_dict = label.to_dict()
    assert set(as_dict.keys()) == {"H", "IH", "SE", "IL", "SI"}

    assert isinstance(probs, dict)
    assert set(probs.keys()) == {"hate_violence", "ideological", "sexual", "illegal", "self_inflicted"}
    for harm, p in probs.items():
        assert set(p.keys()) == {"safe", "topical", "toxic"}
        total = p["safe"] + p["topical"] + p["toxic"]
        assert 0.99 <= total <= 1.01, f"{harm} probs should sum to ~1, got {total}"

