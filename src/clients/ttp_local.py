"""
Local (Transformers) TTP client.

Used for Table 4 rows where the paper runs the TTP prompt on non-OpenAI models
(e.g., Gemma 2 27B).

We reuse the exact ChatML prompt from `prompts/TTP/TTP.txt` and ask the model
to produce the same <Label>{...}</Label> structure, then parse it into HarmLabel.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Literal

from ..utils.taxonomy import HarmLabel

logger = logging.getLogger(__name__)


@dataclass
class LocalTTPResult:
    predicted_label: HarmLabel
    raw_response: Optional[str] = None
    error: Optional[str] = None


class TransformersTTPClient:
    """
    Run the TTP prompt on a local HuggingFace CausalLM.

    This is meant for replication, not production. Defaults to deterministic generation.
    """

    def __init__(
        self,
        model_id: str,
        *,
        prompt_path: str = "prompts/TTP/TTP.txt",
        device: Optional[str] = None,
        dtype: Literal["auto", "float16", "bfloat16"] = "auto",
        quantization: Literal["none", "8bit", "4bit"] = "none",
        max_new_tokens: int = 256,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except ImportError as e:
            raise RuntimeError("Local TTP requires: pip install torch transformers accelerate") from e

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"TTP prompt not found: {prompt_path}")
        self.prompt_template = prompt_file.read_text(encoding="utf-8", errors="replace")
        self._parse_prompt_template()

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

        # dtype handling
        if dtype == "auto":
            torch_dtype = None
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16

        # quantization handling (optional dependency)
        load_kwargs: Dict[str, Any] = {}
        if quantization in {"8bit", "4bit"}:
            try:
                import bitsandbytes  # noqa: F401  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Requested quantization but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                ) from e
            if quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["load_in_4bit"] = True

        logger.info("Loading local TTP model %s on %s (quantization=%s)...", model_id, device, quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            **load_kwargs,
        )
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

    def _parse_prompt_template(self) -> None:
        blocks = re.findall(
            r"<\|im_start\|>(system|user|assistant)\s*(.*?)<\|im_end\|>",
            self.prompt_template,
            re.DOTALL,
        )
        if not blocks:
            raise ValueError("Could not parse any ChatML blocks from prompt")

        template_idx: Optional[int] = None
        for i in range(len(blocks) - 1, -1, -1):
            role, content = blocks[i]
            if role == "user" and "#URL#" in content and "#Body#" in content:
                template_idx = i
                break
        if template_idx is None:
            raise ValueError("Could not find final user template block containing #URL# and #Body#")

        self.user_template = blocks[template_idx][1].strip()

        prefix = []
        for role, content in blocks[:template_idx]:
            role = role.strip().lower()
            content = content.strip()
            if not content:
                continue
            prefix.append((role, content))
        if not prefix or prefix[0][0] != "system":
            raise ValueError("Prompt did not start with a system message block")
        self._prefix = prefix

    def _format_prompt(self, user_message: str) -> str:
        # Represent the ChatML conversation explicitly for a plain-text CausalLM.
        parts = []
        for role, content in self._prefix:
            parts.append(f"{role.upper()}:\n{content}")
        parts.append(f"USER:\n{user_message}")
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> HarmLabel:
        label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL)
        label_str: Optional[str] = None
        if label_match:
            label_str = label_match.group(1)
        else:
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                label_str = stripped
            else:
                for m in re.finditer(r"\{[\s\S]*?\}", content):
                    candidate = m.group(0)
                    if all(k in candidate for k in ["H", "IH", "SE", "IL", "SI"]):
                        label_str = candidate
                        break

        if not label_str:
            raise ValueError(f"Could not find label dict in response: {content[:200]}")

        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, dict):
                label_dict = parsed
            elif isinstance(parsed, str):
                label_dict = json.loads(parsed)
            else:
                raise ValueError(f"Expected dict or string from literal_eval, got {type(parsed)}")
        except (ValueError, SyntaxError):
            label_dict = json.loads(label_str)

        clean: Dict[str, Any] = {}
        for k, v in label_dict.items():
            if isinstance(v, str):
                clean[k] = v.split("-")[0].lower()
            else:
                clean[k] = v
        return HarmLabel.from_dict(clean)

    def evaluate(self, url: str, body: str) -> LocalTTPResult:
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)
        prompt = self._format_prompt(user_message)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            lbl = self._parse_response(text)
            return LocalTTPResult(predicted_label=lbl, raw_response=text)
        except Exception as e:
            return LocalTTPResult(predicted_label=HarmLabel(), raw_response=None, error=str(e))

    def predict(self, text: str) -> HarmLabel:
        r = self.evaluate(url="ttp://text", body=text)
        return r.predicted_label

