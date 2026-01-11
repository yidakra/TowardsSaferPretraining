"""Tests for taxonomy classes."""

import pytest

from src.utils.taxonomy import Dimension, HarmLabel


class TestDimension:
    """Tests for Dimension enum."""

    def test_dimension_values(self):
        """Test dimension enum values."""
        assert Dimension.SAFE.value == "none"
        assert Dimension.TOPICAL.value == "topical"
        assert Dimension.TOXIC.value == "intent"

    @pytest.mark.parametrize("label,expected", [
        ("none", Dimension.SAFE),
        ("topical", Dimension.TOPICAL),
        ("intent", Dimension.TOXIC),
        ("S0", Dimension.SAFE),
        ("S1", Dimension.TOPICAL),
        ("S2", Dimension.TOXIC),
    ])
    def test_from_label(self, label, expected):
        """Test from_label with various valid inputs."""
        assert Dimension.from_label(label) == expected

    def test_from_label_invalid_with_default(self):
        """Test from_label with invalid input and default."""
        assert Dimension.from_label("invalid", default=Dimension.SAFE) == Dimension.SAFE

    def test_from_label_invalid_defaults_to_safe(self):
        """Test from_label with invalid input defaults to SAFE."""
        # Implementation defaults to SAFE when no explicit default is provided
        result = Dimension.from_label("invalid")
        assert result == Dimension.SAFE


class TestHarmLabel:
    """Tests for HarmLabel class."""

    def test_default_label_is_safe(self):
        """Test default HarmLabel is safe."""
        label = HarmLabel()
        assert label.is_safe()
        assert not label.is_toxic()
        assert not label.is_topical()

    def test_toxic_label(self):
        """Test toxic label detection."""
        label = HarmLabel(hate_violence=Dimension.TOXIC)
        assert label.is_toxic()
        assert not label.is_safe()

    def test_topical_label(self):
        """Test topical label detection."""
        label = HarmLabel(sexual=Dimension.TOPICAL)
        assert label.is_topical()
        assert not label.is_safe()
        assert not label.is_toxic()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        label = HarmLabel(
            hate_violence=Dimension.TOXIC,
            sexual=Dimension.TOPICAL
        )
        label_dict = label.to_dict()

        # Abbreviations map to harm categories as defined in HarmCategory.get_short_name_mapping():
        # "H" -> Hate & Violence, "IH" -> Ideological Harm, "SE" -> Sexual,
        # "IL" -> Illegal Activities, "SI" -> Self-Inflicted Harm
        assert label_dict["H"] == "intent"
        assert label_dict["SE"] == "topical"
        assert label_dict["IH"] == "none"
        assert label_dict["IL"] == "none"
        assert label_dict["SI"] == "none"

    def test_mixed_label(self):
        """Test label with both toxic and topical dimensions."""
        label = HarmLabel(
            hate_violence=Dimension.TOXIC,
            sexual=Dimension.TOPICAL
        )
        assert label.is_toxic()  # Has at least one toxic
        assert not label.is_safe()
        # Overall, any toxic dimension should dominate, so it's not "topical".
        assert not label.is_topical()

    def test_multiple_toxic_dimensions(self):
        """Test HarmLabel with multiple toxic dimensions."""
        label = HarmLabel(
            hate_violence=Dimension.TOXIC,
            ideological=Dimension.TOXIC,
            illegal=Dimension.TOXIC
        )
        assert label.is_toxic()
        assert not label.is_safe()
        assert not label.is_topical()
        expected_dict = {
            "H": "intent",
            "IH": "intent",
            "SE": "none",
            "IL": "intent",
            "SI": "none"
        }
        assert label.to_dict() == expected_dict

    def test_multiple_topical_dimensions(self):
        """Test HarmLabel with multiple topical dimensions."""
        label = HarmLabel(
            sexual=Dimension.TOPICAL,
            self_inflicted=Dimension.TOPICAL
        )
        assert label.is_topical()
        assert not label.is_toxic()
        assert not label.is_safe()
        expected_dict = {
            "H": "none",
            "IH": "none",
            "SE": "topical",
            "IL": "none",
            "SI": "topical"
        }
        assert label.to_dict() == expected_dict

    def test_all_dimensions_populated(self):
        """Test HarmLabel with all five dimensions populated."""
        label = HarmLabel(
            hate_violence=Dimension.TOXIC,
            ideological=Dimension.TOPICAL,
            sexual=Dimension.SAFE,
            illegal=Dimension.TOXIC,
            self_inflicted=Dimension.TOPICAL
        )
        assert label.is_toxic()  # Has toxic dimensions
        assert not label.is_safe()
        assert not label.is_topical()  # Toxic dominates
        expected_dict = {
            "H": "intent",
            "IH": "topical",
            "SE": "none",
            "IL": "intent",
            "SI": "topical"
        }
        assert label.to_dict() == expected_dict

    def test_explicit_checks_ih_il_si(self):
        """Test explicit checks on IH, IL, SI dimensions."""
        # Test IH (Ideological Harm) toxic
        label_ih_toxic = HarmLabel(ideological=Dimension.TOXIC)
        assert label_ih_toxic.is_toxic()
        assert not label_ih_toxic.is_topical()
        assert not label_ih_toxic.is_safe()
        assert label_ih_toxic.to_dict()["IH"] == "intent"

        # Test IL (Illegal Activities) topical
        label_il_topical = HarmLabel(illegal=Dimension.TOPICAL)
        assert not label_il_topical.is_toxic()
        assert label_il_topical.is_topical()
        assert not label_il_topical.is_safe()
        assert label_il_topical.to_dict()["IL"] == "topical"

        # Test SI (Self-Inflicted Harm) safe (default)
        label_si_safe = HarmLabel()
        assert not label_si_safe.is_toxic()
        assert not label_si_safe.is_topical()
        assert label_si_safe.is_safe()
        assert label_si_safe.to_dict()["SI"] == "none"