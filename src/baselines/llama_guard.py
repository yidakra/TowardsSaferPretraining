"""
Llama Guard wrapper for content safety classification.

Uses Meta's Llama Guard model for toxicity detection.
"""

import logging
import string
from typing import Optional, List

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    LLAMA_GUARD_AVAILABLE = True
except ImportError:
    LLAMA_GUARD_AVAILABLE = False


class LlamaGuard:
    """
    Wrapper for Meta's Llama Guard safety classifier.

    Llama Guard categories:
    - S1: Violent Crimes
    - S2: Non-Violent Crimes
    - S3: Sex Crimes
    - S4: Child Exploitation
    - S5: Specialized Advice (defamation, privacy, etc.)
    - S6: Indiscriminate Weapons
    - S7: Hate
    - S8: Self-Harm
    - S9: Sexual Content
    - S10: Elections
    - S11: Code Interpreter Abuse

    Maps to our taxonomy:
    - S1, S6, S7 -> Hate/Violence (H)
    - S7, S10 -> Ideological Harm (IH)
    - S3, S4, S9 -> Sexual Content (SE)
    - S2, S5 -> Illegal Activities (IL)
    - S8 -> Self-Inflicted Harm (SI)

    Example:
        guard = LlamaGuard()
        label = guard.predict("This is a test text")
    """

    # Model name on HuggingFace
    MODEL_NAME = "meta-llama/LlamaGuard-7b"

    # Mapping from Llama Guard categories to our taxonomy
    CATEGORY_MAPPING = {
        "S1": ["hate_violence"],          # Violent Crimes
        "S2": ["illegal"],                # Non-Violent Crimes
        "S3": ["sexual"],                 # Sex Crimes
        "S4": ["sexual"],                 # Child Exploitation
        "S5": ["illegal"],                # Specialized Advice
        "S6": ["hate_violence"],          # Indiscriminate Weapons
        "S7": ["hate_violence", "ideological"],  # Hate
        "S8": ["self_inflicted"],         # Self-Harm
        "S9": ["sexual"],                 # Sexual Content
        "S10": ["ideological"],           # Elections
        "S11": ["illegal"],               # Code Interpreter Abuse
    }

    # Prompt template for Llama Guard
    PROMPT_TEMPLATE = string.Template("""<s>[INST] Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Specialized Advice.
S6: Indiscriminate Weapons.
S7: Hate.
S8: Self-Harm.
S9: Sexual Content.
S10: Elections.
S11: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: $text
<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""")

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        use_topical: bool = False
    ):
        """
        Initialize Llama Guard classifier.

        Args:
            model_name: HuggingFace model name
            device: Device (cuda/cpu/mps)
            use_topical: Whether to use topical classification (for backward compatibility)
        """
        if not LLAMA_GUARD_AVAILABLE:
            raise RuntimeError(
                "Llama Guard requires: pip install torch transformers accelerate"
            )

        self.use_topical = use_topical

        logger.info(f"Loading Llama Guard from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        if not torch.cuda.is_available():
            self.model.to(self.device)

        self.model.eval()
        logger.info(f"Llama Guard loaded on {self.device}")

    def classify_text(self, text: str) -> tuple[bool, List[str]]:
        """
        Classify text with Llama Guard.

        Args:
            text: Text to classify

        Returns:
            Tuple of (is_unsafe, violated_categories)
        """
        # Format prompt
        prompt = self.PROMPT_TEMPLATE.substitute(text=text)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        # Parse response
        lines = response.split('\n')
        if not lines:
            logger.warning("Empty response from Llama Guard")
            return False, []

        first_line = lines[0].strip().lower()
        is_unsafe = first_line == "unsafe"

        violated_categories = []
        if is_unsafe and len(lines) > 1:
            # Parse comma-separated categories (e.g., "S1,S7")
            categories_line = lines[1].strip()
            violated_categories = [c.strip() for c in categories_line.split(',')]

        if is_unsafe and not violated_categories:
            logger.warning("Model returned 'unsafe' without category metadata")

        return is_unsafe, violated_categories

    def categories_to_label(
        self,
        violated_categories: List[str]
    ) -> 'HarmLabel':
        """
        Convert Llama Guard categories to HarmLabel.

        Args:
            violated_categories: List of violated category codes (e.g., ["S1", "S7"])

        Returns:
            HarmLabel with classifications
        """
        label = HarmLabel()

        if not violated_categories:
            return label  # All safe

        # Map violated categories to harm dimensions
        harm_flags = {
            "hate_violence": False,
            "ideological": False,
            "sexual": False,
            "illegal": False,
            "self_inflicted": False,
        }

        for category in violated_categories:
            if category in self.CATEGORY_MAPPING:
                for harm_type in self.CATEGORY_MAPPING[category]:
                    harm_flags[harm_type] = True
            else:
                logger.warning(
                    f"Unknown Llama Guard category '{category}' not in CATEGORY_MAPPING. "
                    f"Full violated_categories: {violated_categories}"
                )

        # Set dimensions (Llama Guard only does binary toxic/safe, no topical)
        if harm_flags["hate_violence"]:
            label.hate_violence = Dimension.TOXIC
        if harm_flags["ideological"]:
            label.ideological = Dimension.TOXIC
        if harm_flags["sexual"]:
            label.sexual = Dimension.TOXIC
        if harm_flags["illegal"]:
            label.illegal = Dimension.TOXIC
        if harm_flags["self_inflicted"]:
            label.self_inflicted = Dimension.TOXIC

        return label

    def predict(self, text: str) -> 'HarmLabel':
        """
        Predict harm label for text.

        Args:
            text: Text to classify

        Returns:
            HarmLabel with classifications
        """
        try:
            is_unsafe, violated_categories = self.classify_text(text)
            return self.categories_to_label(violated_categories)
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return HarmLabel()

    def __call__(self, text: str) -> 'HarmLabel':
        """Make classifier callable."""
        return self.predict(text)
