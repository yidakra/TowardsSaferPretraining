"""
OpenAI-backed TTP client.

It's a client/adapter: it produces labels by calling an API.
"""

from __future__ import annotations

import re
import json
import ast
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

from openai import OpenAI  # type: ignore

from ..utils.taxonomy import HarmLabel

logger = logging.getLogger(__name__)


@dataclass
class TTPResult:
    """Result from TTP evaluation."""

    url: str
    body: str
    predicted_label: HarmLabel
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def is_toxic(self) -> bool:
        return self.predicted_label.is_toxic()

    def is_topical(self) -> bool:
        return self.predicted_label.is_topical()

    def is_safe(self) -> bool:
        return self.predicted_label.is_safe()


class OpenAITTPClient:
    """
    Client using the TTP prompt with OpenAI Chat Completions.

    Backwards-compat alias: `TTPEvaluator`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt_path: str = "prompts/TTP/TTP.txt",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"TTP prompt not found: {prompt_path}")

        self.prompt_template = prompt_file.read_text(encoding="utf-8")
        self._parse_prompt_template()

        # Tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.failed_requests = 0

    def _parse_prompt_template(self) -> None:
        """
        Parse ChatML prompt into:
        - `self._prefix_messages`: system + few-shot (user/assistant) examples
        - `self.user_template`: the final user message containing #URL# and #Body#

        This is important for reproducing the paper: the prompt quality relies on
        the included examples (few-shot).
        """
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

        prefix_msgs = []
        for role, content in blocks[:template_idx]:
            role = role.strip()
            content = content.strip()
            if not content:
                continue
            if role not in {"system", "user", "assistant"}:
                continue
            prefix_msgs.append({"role": role, "content": content})

        # Require at least a system message at the front for stable behavior.
        if not prefix_msgs or prefix_msgs[0]["role"] != "system":
            raise ValueError("Prompt did not start with a system message block")

        self._prefix_messages = prefix_msgs
        self.system_message = prefix_msgs[0]["content"]

    def evaluate(self, url: str, body: str) -> TTPResult:
        """Evaluate a single web page (url + body)."""
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)
        last_content: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._prefix_messages + [{"role": "user", "content": user_message}],
                    temperature=self.temperature,
                )

                self.total_requests += 1

                if (
                    response.usage is not None
                    and hasattr(response.usage, "total_tokens")
                    and response.usage.total_tokens is not None
                ):
                    self.total_tokens += response.usage.total_tokens

                content = None
                if (
                    response.choices
                    and len(response.choices) > 0
                    and response.choices[0].message
                    and hasattr(response.choices[0].message, "content")
                ):
                    content = response.choices[0].message.content

                if content is None:
                    response_id = getattr(response, "id", None)
                    response_model = getattr(response, "model", None)
                    raise ValueError(
                        f"No content found in response for URL {url}. "
                        f"Response ID: {response_id}, Model: {response_model}"
                    )

                last_content = content

                label, reasoning = self._parse_response(content)
                return TTPResult(
                    url=url,
                    body=body,
                    predicted_label=label,
                    reasoning=reasoning,
                    raw_response=content,
                )

            except Exception as e:
                logger.warning("Attempt %s failed for %s: %s", attempt + 1, url, e)
                try:
                    msg = str(e)
                    m = re.search(r"try again in\s+([0-9]+\.?[0-9]*)s", msg, re.IGNORECASE)
                    if m:
                        wait_s = float(m.group(1))
                        time.sleep(max(wait_s, self.retry_delay))
                        continue
                except Exception:
                    pass

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.failed_requests += 1
                    fail_open_label = HarmLabel()
                    return TTPResult(
                        url=url,
                        body=body,
                        predicted_label=fail_open_label,
                        raw_response=last_content,
                        error=str(e),
                    )

        raise RuntimeError("Unexpected execution path in evaluate method")

    def predict(self, text: str) -> HarmLabel:
        """Baseline-compatible API: classify a single text and return a HarmLabel."""
        result = self.evaluate(url="ttp://text", body=text)
        return result.predicted_label

    def _parse_response(self, content: str) -> tuple[HarmLabel, Optional[str]]:
        """Parse model response to extract labels and reasoning."""
        reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

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
            try:
                label_dict = json.loads(label_str)
            except json.JSONDecodeError:
                label_dict = {}
                for key in ["H", "IH", "SE", "IL", "SI"]:
                    match = re.search(rf"{key}\s*:\s*([^,}}]+)", label_str)
                    if match:
                        value = match.group(1).strip().strip("\"'")
                        label_dict[key] = value

        clean_dict: Dict[str, Any] = {}
        for key, value in label_dict.items():
            if isinstance(value, str):
                base_value = value.split("-")[0].lower()
                clean_dict[key] = base_value
            else:
                clean_dict[key] = value

        return HarmLabel.from_dict(clean_dict), reasoning

    def evaluate_batch(self, samples: List[tuple[str, str]], show_progress: bool = True) -> List[TTPResult]:
        """Evaluate multiple web pages."""
        results = []
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(samples, desc="Evaluating")
            except ImportError:
                iterator = samples
        else:
            iterator = samples

        for url, body in iterator:
            results.append(self.evaluate(url, body))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
        }


# Backwards compatibility
TTPEvaluator = OpenAITTPClient

