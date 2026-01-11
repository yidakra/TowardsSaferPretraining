"""Tests for taxonomy classes."""

import pytest

from src.utils.taxonomy import Dimension, HarmLabel, HarmCategory


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
        # Toxicity dominance rule: any toxic dimension takes precedence over topical ones.
        # See HarmLabel class docstring for details on the dominance hierarchy.
        assert not label.is_topical()

    def test_toxicity_dominance(self):
        """Test toxicity dominance rule: toxic dimensions override topical ones.

        This test explicitly verifies the design decision documented in HarmLabel docstring
        that any Dimension.TOXIC value takes precedence over Dimension.TOPICAL values.
        See HarmLabel class docstring for details on the dominance hierarchy.
        """
        # Single toxic dimension dominates
        label_single_toxic = HarmLabel(hate_violence=Dimension.TOXIC)
        assert label_single_toxic.is_toxic()
        assert not label_single_toxic.is_topical()
        assert not label_single_toxic.is_safe()

        # Mixed: toxic + topical should be toxic (not topical)
        label_mixed = HarmLabel(
            hate_violence=Dimension.TOXIC,
            sexual=Dimension.TOPICAL
        )
        assert label_mixed.is_toxic()  # Toxic dominates
        assert not label_mixed.is_topical()  # Topical is overridden
        assert not label_mixed.is_safe()

        # Multiple toxic dimensions should be toxic
        label_multi_toxic = HarmLabel(
            hate_violence=Dimension.TOXIC,
            ideological=Dimension.TOXIC
        )
        assert label_multi_toxic.is_toxic()
        assert not label_multi_toxic.is_topical()
        assert not label_multi_toxic.is_safe()

        # Only topical should be topical
        label_only_topical = HarmLabel(sexual=Dimension.TOPICAL)
        assert not label_only_topical.is_toxic()
        assert label_only_topical.is_topical()
        assert not label_only_topical.is_safe()

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
        assert not label.is_topical()  # Toxicity dominance rule
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

    # Edge-case tests for HarmLabel constructor and methods

    def test_invalid_enum_values_strings(self):
        """Test passing invalid string values to HarmLabel constructor.

        Note: dataclass doesn't validate types, so invalid values are accepted
        but cause incorrect method behavior. This documents the current behavior
        and should ideally be fixed with proper type validation.
        """
        # Currently, dataclass accepts invalid strings without raising
        label = HarmLabel(hate_violence="invalid_string")  # type: ignore
        assert label.hate_violence == "invalid_string"
        # Methods behave incorrectly with invalid types
        assert not label.is_safe()  # Should be False because comparison fails
        assert not label.is_toxic()  # Should be False because comparison fails

    def test_invalid_enum_values_ints(self):
        """Test passing integer values to HarmLabel constructor.

        Note: dataclass doesn't validate types, so invalid values are accepted
        but cause incorrect method behavior.
        """
        # Currently, dataclass accepts integers without raising
        label = HarmLabel(hate_violence=42)  # type: ignore
        assert label.hate_violence == 42
        # Methods behave incorrectly with invalid types
        assert not label.is_safe()  # Should be False because comparison fails
        assert not label.is_toxic()  # Should be False because comparison fails

    def test_none_values_not_treated_as_safe(self):
        """Test passing None values to dimensions - currently not treated as Dimension.SAFE.

        Note: This documents current behavior where None values cause incorrect method behavior.
        Ideally, None should be treated as Dimension.SAFE or validation should prevent None.
        """
        # Test individual None values
        label_none_hate = HarmLabel(hate_violence=None)  # type: ignore
        assert label_none_hate.hate_violence is None  # None is stored as-is
        assert not label_none_hate.is_safe()  # Incorrect: should be True but None != Dimension.SAFE
        assert not label_none_hate.is_toxic()  # Correct: None != Dimension.TOXIC

        label_none_sexual = HarmLabel(sexual=None)  # type: ignore
        assert label_none_sexual.sexual is None
        assert not label_none_sexual.is_safe()  # Incorrect behavior

        # Test multiple None values
        label_multi_none = HarmLabel(hate_violence=None, ideological=None, sexual=None)  # type: ignore
        assert label_multi_none.hate_violence is None
        assert label_multi_none.ideological is None
        assert label_multi_none.sexual is None
        assert not label_multi_none.is_safe()  # Incorrect: all None should be treated as safe

    def test_non_dimension_objects(self):
        """Test passing non-Dimension objects - currently no type validation.

        Note: dataclass accepts any object type without validation.
        This documents current behavior and should ideally be fixed.
        """
        # Test with list - accepted without validation
        label_list = HarmLabel(hate_violence=[Dimension.TOXIC])  # type: ignore
        assert label_list.hate_violence == [Dimension.TOXIC]
        assert not label_list.is_safe()  # Incorrect behavior

        # Test with dict - accepted without validation
        label_dict = HarmLabel(sexual={"value": "toxic"})  # type: ignore
        assert label_dict.sexual == {"value": "toxic"}
        assert not label_dict.is_safe()  # Incorrect behavior

        # Test with custom object - accepted without validation
        class CustomObject:
            pass

        label_custom = HarmLabel(illegal=CustomObject())  # type: ignore
        assert isinstance(label_custom.illegal, CustomObject)
        assert not label_custom.is_safe()  # Incorrect behavior

    def test_equality_and_str_repr(self):
        """Test equality and string representation if implemented."""
        # Test equality (dataclass provides __eq__)
        label1 = HarmLabel(hate_violence=Dimension.TOXIC, sexual=Dimension.TOPICAL)
        label2 = HarmLabel(hate_violence=Dimension.TOXIC, sexual=Dimension.TOPICAL)
        label3 = HarmLabel(hate_violence=Dimension.TOPICAL, sexual=Dimension.TOXIC)

        assert label1 == label2
        assert label1 != label3

        # Test string representation
        str_repr = str(label1)
        assert "HarmLabel" in str_repr
        assert "hate_violence=<Dimension.TOXIC:" in str_repr
        assert "sexual=<Dimension.TOPICAL:" in str_repr

    def test_methods_with_none_inputs(self):
        """Test that methods behave consistently when None values are passed to constructor.

        Note: Current behavior is inconsistent - None values cause methods to behave incorrectly.
        Ideally, None should be treated as Dimension.SAFE or validation should prevent None.
        """
        # Methods behave incorrectly with None values
        label_none = HarmLabel(hate_violence=None, sexual=None)  # type: ignore

        assert not label_none.is_safe()  # Incorrect: should be True
        assert not label_none.is_toxic()  # Correct: None != Dimension.TOXIC
        assert not label_none.is_topical()  # Correct: None != Dimension.TOPICAL

        # to_dict will fail because None doesn't have a .value attribute
        with pytest.raises(AttributeError):
            label_none.to_dict()

        # get_toxic_harms succeeds and returns empty list (None != Dimension.TOXIC)
        assert label_none.get_toxic_harms() == []

    def test_methods_with_invalid_construction_fallback(self):
        """Test method behavior when object is constructed with invalid inputs.

        Note: This test documents what happens with invalid inputs vs valid Dimension enums.
        """
        # Create a label with valid Dimension enum - should work correctly
        label_toxic = HarmLabel(hate_violence=Dimension.TOXIC)
        assert label_toxic.is_toxic()
        assert not label_toxic.is_safe()
        assert not label_toxic.is_topical()

        # Test to_dict with toxic label
        toxic_dict = label_toxic.to_dict()
        assert toxic_dict["H"] == "intent"
        assert toxic_dict["IH"] == "none"

        # Test get_toxic_harms
        toxic_harms = label_toxic.get_toxic_harms()
        assert len(toxic_harms) == 1
        assert HarmCategory.HATE_VIOLENCE in toxic_harms

        # Compare with invalid input - to_dict fails, get_toxic_harms succeeds
        label_invalid = HarmLabel(hate_violence="invalid")  # type: ignore
        with pytest.raises(AttributeError):
            label_invalid.to_dict()
        # get_toxic_harms succeeds with invalid input (string != Dimension.TOXIC)
        assert label_invalid.get_toxic_harms() == []