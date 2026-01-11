import os

import pytest


pytestmark = [pytest.mark.requires_model, pytest.mark.slow]


if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Integration test (downloads model). Set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("test_text", [
    "This is a neutral text about cooking and gardening.",  # original neutral text
    "",  # empty string
    "Hello world! " * 1000,  # very long repeated text
    "Text with emojis 🚀🔥 and special chars: áéíóúñ 中文 русский",  # unicode/special chars
])
def test_harmformer_predict_and_probabilities_smoke(test_text):
    # Arrange
    from src.models import HarmFormer

    model = HarmFormer()

    # Act & Assert - no exceptions should be raised
    label = model.predict(test_text)
    probs = model.get_probabilities(test_text)

    # Assert label return type and structure
    assert hasattr(label, "to_dict")
    as_dict = label.to_dict()
    assert set(as_dict.keys()) == {"H", "IH", "SE", "IL", "SI"}
    # Values should be valid dimension strings
    valid_dimensions = {"none", "topical", "intent"}
    for key, value in as_dict.items():
        assert value in valid_dimensions, f"Invalid dimension '{value}' for key '{key}'"

    # Assert probabilities return type and structure
    assert isinstance(probs, dict)
    assert set(probs.keys()) == {"hate_violence", "ideological", "sexual", "illegal", "self_inflicted"}
    for harm, p in probs.items():
        assert set(p.keys()) == {"safe", "topical", "toxic"}
        assert all(isinstance(v, float) for v in p.values()), f"Probabilities for {harm} should be floats"
        total = p["safe"] + p["topical"] + p["toxic"]
        assert total == pytest.approx(1, rel=0.01)

