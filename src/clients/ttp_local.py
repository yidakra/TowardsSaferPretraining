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
        # Allow more tokens for models that produce verbose reasoning before the label
        # R1 models and reasoning models need 1500+ tokens to complete their chain-of-thought
        max_new_tokens: int = 1536,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except ImportError as e:
            raise RuntimeError("Local TTP requires: pip install torch transformers accelerate") from e

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        # Cap input length to avoid KV-cache OOM on large prompts.
        # Override via env if you want to force a smaller cap.
        self.max_input_tokens: Optional[int] = None
        env_max_in = os.environ.get("TTP_LOCAL_MAX_INPUT_TOKENS")
        if env_max_in and env_max_in.strip():
            try:
                self.max_input_tokens = int(env_max_in)
            except Exception:
                self.max_input_tokens = None
        env_max_out = os.environ.get("TTP_LOCAL_MAX_NEW_TOKENS")
        if env_max_out and env_max_out.strip():
            try:
                self.max_new_tokens = int(env_max_out)
            except Exception:
                pass

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

        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        auth_kwargs: Dict[str, Any] = {"token": hf_token} if hf_token else {}
        local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY") or os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")
        local_files_only = str(local_files_only).strip().lower() in {"1", "true", "yes", "on"}

        env_use_fast = os.environ.get("TTP_LOCAL_USE_FAST")
        if env_use_fast is not None:
            use_fast = str(env_use_fast).strip().lower() in {"1", "true", "yes", "on"}
        else:
            use_fast = True
            try:
                import tokenizers  # type: ignore  # noqa: F401
            except Exception:
                use_fast = False
                logger.warning("Fast tokenizer unavailable; falling back to slow tokenizer.")

        trust_remote_code = os.environ.get("TTP_LOCAL_TRUST_REMOTE_CODE")
        trust_remote_code = str(trust_remote_code).strip().lower() in {"1", "true", "yes", "on"}

        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **auth_kwargs,
        )
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

        # dtype handling
        is_gemma = "gemma" in model_id.lower()
        if dtype == "auto":
            # On CUDA, defaulting to fp32 is often too large for 27B/32B models.
            # Gemma 2 is numerically more stable in bfloat16 on A100.
            if device == "cuda":
                torch_dtype = torch.bfloat16 if is_gemma else torch.float16
            else:
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
        if device == "cuda" and torch.cuda.device_count() > 1:
            # Encourage sharding across multiple GPUs instead of filling GPU:0.
            # Snellius A100 nodes are often 40GB per GPU; leave some headroom for KV cache.
            load_kwargs.setdefault("max_memory", {i: "38GiB" for i in range(torch.cuda.device_count())})
            load_kwargs["max_memory"].setdefault("cpu", "64GiB")
            device_map = "balanced"
        else:
            device_map = "auto" if device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **auth_kwargs,
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
        model_id_lower = self.model_id.lower()
        is_gemma = "gemma" in model_id_lower
        is_gpt_oss = "gpt-oss" in model_id_lower

        # Prefer the model's chat template when available (important for instruct models like Gemma / LLaMA).
        if hasattr(self.tokenizer, "apply_chat_template"):
            # For Gemma models, use a SIMPLIFIED prompt without the verbose few-shot examples
            # Gemma 2 seems to struggle with very long prompts
            if is_gemma:
                try:
                    # Get only the system message (taxonomy definition)
                    system_content = ""
                    for role, content in self._prefix:
                        if role == "system":
                            # Truncate system message to key parts for Gemma
                            system_content = content[:8000]  # Keep taxonomy but limit length
                            break

                    # Create a more focused prompt for Gemma
                    combined_content = f"""You are an expert content moderator. Classify the following web content.

{system_content[:6000]}

Based on these guidelines, analyze the content and output ONLY a JSON label in this exact format:
<Label>{{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}}</Label>

Content to classify:
{user_message}

Output the label now:"""

                    messages = [{"role": "user", "content": combined_content}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    print(f"[DEBUG] Using simplified Gemma format for {self.model_id}", file=__import__('sys').stderr)
                    return formatted
                except Exception as e:
                    print(f"[DEBUG] Gemma chat template failed: {e}", file=__import__('sys').stderr)
            elif is_gpt_oss:
                try:
                    system_content = ""
                    for role, content in self._prefix:
                        if role == "system":
                            system_content = content[:4000]
                            break

                    combined_content = f"""You are an expert content moderator. Follow the taxonomy below and output ONLY the label.

{system_content}

Output ONLY in this exact format:
<Label>{{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}}</Label>

Content to classify:
{user_message}
"""

                    messages = [{"role": "user", "content": combined_content}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    print(f"[DEBUG] Using simplified GPT-OSS format for {self.model_id}", file=__import__('sys').stderr)
                    return formatted
                except Exception as e:
                    print(f"[DEBUG] GPT-OSS chat template failed: {e}", file=__import__('sys').stderr)
            else:
                # Try a couple role conventions: many models use "assistant", some use "model".
                for map_assistant_to_model in (False, True):
                    try:
                        messages = []
                        for role, content in self._prefix:
                            r = "model" if (map_assistant_to_model and role == "assistant") else role
                            messages.append({"role": r, "content": content})
                        messages.append({"role": "user", "content": user_message})
                        formatted = self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        # Debug: print the formatted prompt for troubleshooting
                        print(f"[DEBUG] Using chat template for {self.model_id}, map_assistant={map_assistant_to_model}", file=__import__('sys').stderr)
                        return formatted
                    except Exception as e:
                        print(f"[DEBUG] Chat template failed for {self.model_id}, map_assistant={map_assistant_to_model}: {e}", file=__import__('sys').stderr)
                        continue

        # Fallback: Use Gemma turn markers if it's Gemma, otherwise plain text format
        if is_gemma:
            # Gemma uses <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n format
            # Use simplified prompt
            system_content = ""
            for role, content in self._prefix:
                if role == "system":
                    system_content = content[:6000]
                    break
            combined = f"""You are a content moderator. Classify this content.

Guidelines (summarized): {system_content[:4000]}

Content:
{user_message}

Output ONLY: <Label>{{H: ..., IH: ..., SE: ..., IL: ..., SI: ...}}</Label>"""

            formatted = f"<start_of_turn>user\n{combined}<end_of_turn>\n<start_of_turn>model\n"
            print(f"[DEBUG] Using Gemma fallback turn format for {self.model_id}", file=__import__('sys').stderr)
            return formatted

        # Generic fallback for other models
        parts = []
        for role, content in self._prefix:
            parts.append(f"{role.upper()}:\n{content}")
        parts.append(f"USER:\n{user_message}")
        print(f"[DEBUG] Using fallback prompt format for {self.model_id}", file=__import__('sys').stderr)
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> HarmLabel:
        content = content.strip()

        # Strip fenced code blocks if present (common in reasoning models)
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.IGNORECASE)
        if fence:
            content = fence.group(1).strip()

        # Handle R1/reasoning model output - they use <think>...</think> tags
        # Extract content after </think> if present
        think_end = re.search(r"</think>", content, re.IGNORECASE)
        if think_end:
            content_after_think = content[think_end.end():].strip()
            if content_after_think:
                logger.info(f"Found </think> tag, using content after: {content_after_think[:200]}")
                content = content_after_think

        # First try: exact <Label> tag format
        label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL | re.IGNORECASE)
        label_str: Optional[str] = None
        label_list: Optional[list] = None
        if label_match:
            label_str = label_match.group(1)
        else:
            # Also search in full original content in case label is within thinking
            if not label_str:
                full_label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL | re.IGNORECASE)
                if full_label_match:
                    label_str = full_label_match.group(1)

            # Second try: look for any JSON-like dict containing the required keys
            if not label_str:
                for m in re.finditer(r"\{[\s\S]*?\}", content):
                    candidate = m.group(0)
                    if all(k in candidate for k in ["H", "IH", "SE", "IL", "SI"]):
                        label_str = candidate
                        break

            # Try a list-form label like: ['none','topical','intent','none','none']
            if not label_str:
                for m in re.finditer(r"\[[\s\S]*?\]", content):
                    candidate = m.group(0)
                    try:
                        parsed_list = ast.literal_eval(candidate)
                        if isinstance(parsed_list, list) and len(parsed_list) == 5:
                            label_list = parsed_list
                            break
                    except Exception:
                        continue

            # Third try: if content starts with a brace and contains the keys
            if not label_str and content.startswith("{") and all(k in content for k in ["H", "IH", "SE", "IL", "SI"]):
                # Find the closing brace
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    label_str = content[:end_pos]

        if label_list is not None:
            return HarmLabel.from_list([str(x) for x in label_list])

        if not label_str:
            # Last resort: try to extract from natural language response
            # Look for patterns like "H: None", "IH: Intent-1", etc.
            extracted = {}
            for key in ["H", "IH", "SE", "IL", "SI"]:
                pattern = rf"{key}\s*:\s*([^{{}}\n,]*)"
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"\']')
                    if value:
                        extracted[key] = value.split("-")[0].lower()
            if extracted and len(extracted) >= 3:  # At least 3 keys found
                label_str = str(extracted).replace("'", '"')

        if not label_str:
            raise ValueError(f"Could not find label dict in response: {content[:300]}")

        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, dict):
                label_dict = parsed
            elif isinstance(parsed, str):
                label_dict = json.loads(parsed)
            else:
                raise ValueError(f"Expected dict or string from literal_eval, got {type(parsed)}")
        except (ValueError, SyntaxError, json.JSONDecodeError):
            # Try to fix common issues
            label_str = label_str.replace("'", '"')  # Replace single quotes with double
            try:
                label_dict = json.loads(label_str)
            except json.JSONDecodeError:
                # Try to parse individual key-value pairs
                label_dict = {}
                for key in ["H", "IH", "SE", "IL", "SI"]:
                    match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', label_str)
                    if match:
                        value = match.group(1).split("-")[0].lower()
                        label_dict[key] = value
                    else:
                        match = re.search(rf"{key}\s*:\s*([^,}}]+)", label_str)
                        if match:
                            value = match.group(1).strip().strip('"\']').split("-")[0].lower()
                            label_dict[key] = value

        if not label_dict:
            raise ValueError(f"Failed to parse label dict from: {label_str}")

        clean: Dict[str, Any] = {}
        for k, v in label_dict.items():
            if isinstance(v, str):
                clean[k] = v.split("-")[0].lower()
            else:
                clean[k] = str(v).lower()
        return HarmLabel.from_dict(clean)

    def evaluate(self, url: str, body: str) -> LocalTTPResult:
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)

        # Model-specific prompt adjustments
        is_r1 = "r1" in self.model_id.lower() or "deepseek" in self.model_id.lower()

        if is_r1:
            # R1 models will reason anyway, so tell them to put the label at the end
            user_message = (
                user_message
                + "\n\n# OUTPUT FORMAT\n"
                + "Think through your analysis step by step, then at the END of your response, "
                + "output the final label in this exact format:\n"
                + "<Label>{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}</Label>\n"
                + "The <Label>...</Label> tags are REQUIRED at the end."
            )
        else:
            # For other models, ask for concise output
            user_message = (
                user_message
                + "\n\n# OUTPUT FORMAT\n"
                + "You must respond with ONLY the label in this exact format:\n"
                + "<Label>{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}</Label>\n"
                + "Do not include any other text, reasoning, or explanation."
            )

        prompt = self._format_prompt(user_message)

        try:
            max_len = self.max_input_tokens
            if max_len is None:
                # tokenizer.model_max_length is sometimes a huge sentinel; clamp to something reasonable by default.
                tmax = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
                max_len = tmax if 0 < tmax <= 32768 else 8192

            print(f"[DEBUG] Prompt length: {len(prompt)} chars, max_input_tokens: {max_len}", file=__import__('sys').stderr)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_len))
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_token_count = inputs["input_ids"].shape[-1]
            print(f"[DEBUG] Tokenized to {input_token_count} tokens", file=__import__('sys').stderr)

            # Model-specific generation parameters
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "min_new_tokens": 16,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Gemma models need special handling - they can produce empty output with certain settings
            is_gemma = "gemma" in self.model_id.lower()
            is_r1 = "r1" in self.model_id.lower() or "deepseek" in self.model_id.lower()

            if is_gemma:
                # Gemma 2 instruction models have specific requirements
                gen_kwargs["num_beams"] = 1
                gen_kwargs["early_stopping"] = False
                gen_kwargs["do_sample"] = False
                gen_kwargs["min_new_tokens"] = 1
                # Remove temperature for Gemma (it can cause issues with do_sample=False)
                gen_kwargs.pop("temperature", None)
                # Ensure we have proper attention mask for Gemma
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = self._torch.ones_like(inputs["input_ids"])
                # For Gemma, avoid generating pad tokens; set pad to eos and ban pad in outputs.
                if self.tokenizer.eos_token_id is not None:
                    gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
                if self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                    gen_kwargs["bad_words_ids"] = [[self.tokenizer.pad_token_id]]
                # Allow stopping on EOS or <end_of_turn> if available.
                eos_ids = []
                if self.tokenizer.eos_token_id is not None:
                    eos_ids.append(self.tokenizer.eos_token_id)
                try:
                    eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    if eot_id is not None and eot_id >= 0:
                        eos_ids.append(int(eot_id))
                except Exception:
                    pass
                if eos_ids:
                    gen_kwargs["eos_token_id"] = list(dict.fromkeys(eos_ids))
            elif is_r1:
                # R1 reasoning models need more tokens and work better with specific settings
                gen_kwargs["max_new_tokens"] = max(self.max_new_tokens, 2048)
                gen_kwargs.pop("temperature", None)
            else:
                gen_kwargs["temperature"] = 0.0

            # Debug: print generation kwargs
            print(f"[DEBUG] Generation kwargs: {gen_kwargs}", file=__import__('sys').stderr)

            with self._torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)

            generated = out[0][inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Debug: show first 300 chars of raw output
            if not text and is_gemma:
                # Retry once with mild sampling for Gemma if output is empty.
                try:
                    retry_kwargs = dict(gen_kwargs)
                    retry_kwargs["do_sample"] = True
                    retry_kwargs["temperature"] = 0.2
                    retry_kwargs["top_p"] = 0.9
                    retry_kwargs["min_new_tokens"] = 1
                    print("[DEBUG] Gemma empty output; retrying with sampling...", file=__import__('sys').stderr)
                    out = self.model.generate(**inputs, **retry_kwargs)
                    generated = out[0][inputs["input_ids"].shape[-1] :]
                    text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                except Exception as retry_e:
                    print(f"[DEBUG] Gemma retry failed: {retry_e}", file=__import__('sys').stderr)

            if not text:
                print(f"[DEBUG] WARNING: Empty output from {self.model_id}!", file=__import__('sys').stderr)
                # Try decoding without skipping special tokens to see what's there
                raw_text = self.tokenizer.decode(generated, skip_special_tokens=False)
                print(f"[DEBUG] Raw tokens (with special): {raw_text[:300]!r}", file=__import__('sys').stderr)
                print(f"[DEBUG] Generated token ids (first 50): {generated[:50].tolist()}", file=__import__('sys').stderr)
                print(f"[DEBUG] Total generated tokens: {len(generated)}", file=__import__('sys').stderr)
            print(f"[DEBUG] Generated {len(text)} chars of output", file=__import__('sys').stderr)
            try:
                lbl = self._parse_response(text)
            except Exception as e:
                print(f"[DEBUG] Failed to parse TTP label from {self.model_id}. Raw output: {text[:500]!r}", file=__import__('sys').stderr)
                raise RuntimeError(f"Failed to parse TTP label. output_head={text[:240]!r}. error={e}") from e
            return LocalTTPResult(predicted_label=lbl, raw_response=text)
        except Exception as e:
            return LocalTTPResult(predicted_label=HarmLabel(), raw_response=None, error=str(e))

    def predict(self, text: str) -> HarmLabel:
        r = self.evaluate(url="ttp://text", body=text)
        # Do not silently fail-open: callers (e.g. evaluation scripts) should count failures.
        if r.error:
            raise RuntimeError(r.error)
        return r.predicted_label

