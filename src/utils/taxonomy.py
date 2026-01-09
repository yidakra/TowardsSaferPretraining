"""
Taxonomy definitions for the three-dimensional safety classification framework.
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class HarmCategory(Enum):
    """Five harm categories in the taxonomy."""
    HATE_VIOLENCE = "Hate & Violence"
    IDEOLOGICAL = "Ideological Harm"
    SEXUAL = "Sexual"
    ILLEGAL = "Illegal Activities"
    SELF_INFLICTED = "Self-Inflicted Harm"

    @classmethod
    def get_short_names(cls) -> Dict[str, str]:
        """Get mapping from full names to short names."""
        return {
            cls.HATE_VIOLENCE.value: "H",
            cls.IDEOLOGICAL.value: "IH",
            cls.SEXUAL.value: "SE",
            cls.ILLEGAL.value: "IL",
            cls.SELF_INFLICTED.value: "SI",
        }

    @classmethod
    def from_short_name(cls, short_name: str) -> "HarmCategory":
        """Get HarmCategory from short name."""
        mapping = {
            "H": cls.HATE_VIOLENCE,
            "IH": cls.IDEOLOGICAL,
            "SE": cls.SEXUAL,
            "IL": cls.ILLEGAL,
            "SI": cls.SELF_INFLICTED,
        }
        return mapping[short_name]


class Dimension(Enum):
    """Three dimensions in the taxonomy."""
    SAFE = "none"  # Also called "None" in the data
    TOPICAL = "topical"
    TOXIC = "intent"  # Also called "Intent" in the data (internally uses "intent")

    @classmethod
    def from_label(cls, label: str, default: "Dimension" = None) -> "Dimension":
        """
        Parse dimension from various label formats.

        Args:
            label: Label string to parse
            default: Default dimension to return if parsing fails (default: SAFE)

        Returns:
            Dimension enum value
        """
        if default is None:
            default = cls.SAFE

        label_lower = label.lower().strip()
        if label_lower in ["none", "safe", ""]:
            return cls.SAFE
        elif label_lower == "topical":
            return cls.TOPICAL
        elif label_lower in ["intent", "toxic"]:
            return cls.TOXIC
        else:
            # Handle unexpected values gracefully
            print(f"Warning: Unknown dimension label '{label}', defaulting to {default.value}")
            return default


@dataclass
class HarmLabel:
    """Represents harm labels across all categories."""
    hate_violence: Dimension = Dimension.SAFE
    ideological: Dimension = Dimension.SAFE
    sexual: Dimension = Dimension.SAFE
    illegal: Dimension = Dimension.SAFE
    self_inflicted: Dimension = Dimension.SAFE

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "H": self.hate_violence.value,
            "IH": self.ideological.value,
            "SE": self.sexual.value,
            "IL": self.illegal.value,
            "SI": self.self_inflicted.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "HarmLabel":
        """Create from dictionary with short names."""
        return cls(
            hate_violence=Dimension.from_label(data.get("H", "none")),
            ideological=Dimension.from_label(data.get("IH", "none")),
            sexual=Dimension.from_label(data.get("SE", "none")),
            illegal=Dimension.from_label(data.get("IL", "none")),
            self_inflicted=Dimension.from_label(data.get("SI", "none")),
        )

    @classmethod
    def from_list(cls, labels: List[str]) -> "HarmLabel":
        """Create from list format: [H, IH, SE, IL, SI]."""
        if len(labels) != 5:
            raise ValueError(f"Expected 5 labels, got {len(labels)}")
        return cls(
            hate_violence=Dimension.from_label(labels[0]),
            ideological=Dimension.from_label(labels[1]),
            sexual=Dimension.from_label(labels[2]),
            illegal=Dimension.from_label(labels[3]),
            self_inflicted=Dimension.from_label(labels[4]),
        )

    def is_toxic(self) -> bool:
        """Check if any harm category is toxic."""
        return any([
            self.hate_violence == Dimension.TOXIC,
            self.ideological == Dimension.TOXIC,
            self.sexual == Dimension.TOXIC,
            self.illegal == Dimension.TOXIC,
            self.self_inflicted == Dimension.TOXIC,
        ])

    def is_topical(self) -> bool:
        """Check if any harm category is topical (but not toxic)."""
        has_topical = any([
            self.hate_violence == Dimension.TOPICAL,
            self.ideological == Dimension.TOPICAL,
            self.sexual == Dimension.TOPICAL,
            self.illegal == Dimension.TOPICAL,
            self.self_inflicted == Dimension.TOPICAL,
        ])
        return has_topical and not self.is_toxic()

    def is_safe(self) -> bool:
        """Check if all harm categories are safe."""
        return all([
            self.hate_violence == Dimension.SAFE,
            self.ideological == Dimension.SAFE,
            self.sexual == Dimension.SAFE,
            self.illegal == Dimension.SAFE,
            self.self_inflicted == Dimension.SAFE,
        ])

    def get_toxic_harms(self) -> List[HarmCategory]:
        """Get list of harm categories that are toxic."""
        toxic_harms = []
        mapping = [
            (self.hate_violence, HarmCategory.HATE_VIOLENCE),
            (self.ideological, HarmCategory.IDEOLOGICAL),
            (self.sexual, HarmCategory.SEXUAL),
            (self.illegal, HarmCategory.ILLEGAL),
            (self.self_inflicted, HarmCategory.SELF_INFLICTED),
        ]
        for dim, harm in mapping:
            if dim == Dimension.TOXIC:
                toxic_harms.append(harm)
        return toxic_harms
