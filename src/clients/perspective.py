"""
Perspective Comment Analyzer client adapter.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import os
from typing import Optional, Dict, Any, Literal, List

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

try:
    from googleapiclient import discovery  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False


class PerspectiveClient:
    """
    Adapter for Google's Perspective API.

    Maps Perspective toxicity scores to our taxonomy using threshold-based classification.
    """

    ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
        "SEXUALLY_EXPLICIT",
    ]

    # Paper-faithful defaults (Mendu et al. 2025, Table 4)
    PAPER_TABLE4_CHUNK_CHARS = 500
    PAPER_TABLE4_THRESHOLD = 0.4

    # Legacy taxonomy mapping defaults (kept for non-Table-4 usage)
    TOXIC_THRESHOLD = 0.7
    TOPICAL_THRESHOLD = 0.3

    def __init__(
        self,
        api_key: str,
        mode: Literal["paper_table4", "taxonomy"] = "paper_table4",
        # Only used in taxonomy mode
        toxic_threshold: float = TOXIC_THRESHOLD,
        topical_threshold: float = TOPICAL_THRESHOLD,
        # Only used in paper_table4 mode
        paper_threshold: float = PAPER_TABLE4_THRESHOLD,
        paper_chunk_chars: int = PAPER_TABLE4_CHUNK_CHARS,
        # Throttling to avoid 429s (Perspective quotas are often low).
        # Set via env PERSPECTIVE_MIN_INTERVAL_S to override default.
        min_interval_s: Optional[float] = None,
        languages: Optional[list] = None,
    ):
        if not PERSPECTIVE_AVAILABLE:
            raise RuntimeError("Perspective API requires: pip install google-api-python-client")

        self.mode = mode
        self.toxic_threshold = toxic_threshold
        self.topical_threshold = topical_threshold
        self.paper_threshold = paper_threshold
        self.paper_chunk_chars = paper_chunk_chars
        self.languages = languages if languages is not None else ["en"]

        self.api_key = api_key
        self.client = None

        if min_interval_s is None:
            env = os.environ.get("PERSPECTIVE_MIN_INTERVAL_S")
            try:
                min_interval_s = float(env) if env is not None and env.strip() else 0.35
            except Exception:
                min_interval_s = 0.35
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._last_request_ts: float = 0.0

        # Prefer googleapiclient when it works, but fall back to direct REST calls.
        # Some key configurations block access to the discovery document endpoint even
        # when the actual `comments:analyze` method works.
        try:
            self.client = discovery.build(  # type: ignore
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                static_discovery=True,
            )
            logger.info("Initialized Perspective API client (googleapiclient)")
        except Exception as e:
            logger.warning(
                "Failed to initialize Perspective API client via googleapiclient; "
                "falling back to direct REST. Error: %s",
                e,
            )

    def _analyze_via_rest(self, analyze_request: dict) -> Optional[dict]:
        url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_key}"
        data = json.dumps(analyze_request).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def _analyze(self, text: str, requested_attributes: List[str], max_retries: int = 3) -> Optional[Dict[str, float]]:
        text = text.strip()
        if not text:
            logger.warning("Empty text provided to Perspective API")
            return None
        if len(text) > 20000:
            logger.warning("Text too long (%s chars), truncating to 20000", len(text))
            text = text[:20000]

        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {attr: {} for attr in requested_attributes},
            "languages": self.languages,
        }

        for attempt in range(max_retries):
            try:
                # Best-effort throttling to avoid repeated 429s, especially with chunking.
                if self.min_interval_s > 0:
                    now = time.time()
                    sleep_for = (self._last_request_ts + self.min_interval_s) - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                if self.client is not None:
                    response = self.client.comments().analyze(body=analyze_request).execute()
                else:
                    response = self._analyze_via_rest(analyze_request)
                self._last_request_ts = time.time()

                scores = {}
                for attr in requested_attributes:
                    if attr in response["attributeScores"]:
                        score = response["attributeScores"][attr]["summaryScore"]["value"]
                        scores[attr] = score
                return scores
            except HttpError as e:  # type: ignore
                if e.resp.status == 429 or (500 <= e.resp.status < 600):
                    wait_time = 2**attempt
                    logger.warning(
                        "Rate limited or server error (%s), waiting %ss...",
                        e.resp.status,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                logger.error("Perspective API error: %s", e)
                return None
            except urllib.error.HTTPError as e:
                # REST fallback: handle rate limits / transient errors similarly
                status = getattr(e, "code", None)
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    retry_after = None
                    try:
                        ra = e.headers.get("Retry-After") if getattr(e, "headers", None) is not None else None
                        if ra:
                            retry_after = float(ra)
                    except Exception:
                        retry_after = None
                    wait_time = retry_after if retry_after is not None else (2**attempt)
                    logger.warning("Perspective REST HTTPError (%s), waiting %ss...", status, wait_time)
                    time.sleep(wait_time)
                    continue
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                logger.error("Perspective REST HTTPError: %s body=%s", status, body[:200])
                return None
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                return None

        logger.error("Failed after %s retries", max_retries)
        return None

    def get_scores(self, text: str, max_retries: int = 3) -> Optional[Dict[str, float]]:
        """Get multiple Perspective attribute scores (legacy taxonomy mode)."""
        return self._analyze(text=text, requested_attributes=self.ATTRIBUTES, max_retries=max_retries)

    def get_toxicity_score_paper_table4(
        self,
        text: str,
        *,
        chunk_chars: Optional[int] = None,
        max_retries: int = 3,
    ) -> Optional[float]:
        """
        Paper-faithful Perspective setup (Table 4):
        - Split long text into fixed-size character chunks (default 500 chars)
        - Query TOXICITY for each chunk
        - Aggregate by max score across chunks
        """
        chunk_chars = self.paper_chunk_chars if chunk_chars is None else chunk_chars
        text = (text or "").strip()
        if not text:
            return None

        # Chunking to cope with longer texts; paper uses 500 characters.
        chunks: List[str]
        if chunk_chars and len(text) > chunk_chars:
            chunks = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        else:
            chunks = [text]

        max_score: Optional[float] = None
        for ch in chunks:
            scores = self._analyze(text=ch, requested_attributes=["TOXICITY"], max_retries=max_retries)
            if not scores or "TOXICITY" not in scores:
                continue
            s = float(scores["TOXICITY"])
            if (max_score is None) or (s > max_score):
                max_score = s

        return max_score

    def scores_to_label(self, scores: dict) -> HarmLabel:
        label = HarmLabel()

        general_toxicity = max(
            [
                scores.get("TOXICITY", 0.0),
                scores.get("SEVERE_TOXICITY", 0.0),
                scores.get("INSULT", 0.0),
                scores.get("PROFANITY", 0.0),
            ]
        )

        if general_toxicity >= self.toxic_threshold:
            label.hate_violence = Dimension.TOXIC
        elif general_toxicity >= self.topical_threshold:
            label.hate_violence = Dimension.TOPICAL

        identity_attack = scores.get("IDENTITY_ATTACK", 0.0)
        if identity_attack >= self.toxic_threshold:
            label.ideological = Dimension.TOXIC
        elif identity_attack >= self.topical_threshold:
            label.ideological = Dimension.TOPICAL

        sexually_explicit = scores.get("SEXUALLY_EXPLICIT", 0.0)
        if sexually_explicit >= self.toxic_threshold:
            label.sexual = Dimension.TOXIC
        elif sexually_explicit >= self.topical_threshold:
            label.sexual = Dimension.TOPICAL

        threat = scores.get("THREAT", 0.0)
        threat_severity = Dimension.SAFE
        if threat >= self.toxic_threshold:
            threat_severity = Dimension.TOXIC
        elif threat >= self.topical_threshold:
            threat_severity = Dimension.TOPICAL

        severity_order = {Dimension.SAFE: 0, Dimension.TOPICAL: 1, Dimension.TOXIC: 2}
        if severity_order[threat_severity] > severity_order[label.hate_violence]:
            label.hate_violence = threat_severity

        return label

    def predict(self, text: str) -> HarmLabel:
        if self.mode == "paper_table4":
            score = self.get_toxicity_score_paper_table4(text)
            if score is None:
                logger.warning("Failed to get Perspective toxicity score, returning safe label")
                return HarmLabel()

            is_toxic = score >= float(self.paper_threshold)
            if not is_toxic:
                return HarmLabel()

            # Represent binary "toxic dimension" decision in our taxonomy by marking all harms toxic.
            # This keeps per-harm metrics interpretable (they match the overall toxic decision).
            return HarmLabel(
                hate_violence=Dimension.TOXIC,
                ideological=Dimension.TOXIC,
                sexual=Dimension.TOXIC,
                illegal=Dimension.TOXIC,
                self_inflicted=Dimension.TOXIC,
            )

        scores = self.get_scores(text)
        if scores is None:
            logger.warning("Failed to get scores, returning safe label")
            return HarmLabel()
        return self.scores_to_label(scores)

    def __call__(self, text: str) -> HarmLabel:
        return self.predict(text)


# Backwards compatibility
PerspectiveAPI = PerspectiveClient

