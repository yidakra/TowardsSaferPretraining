"""
Optional Weights & Biases integration helpers.

Design goals for this repro repo:
- stay fail-open when wandb is unavailable or disabled
- keep script code small and consistent
- avoid logging secrets from CLI/env configuration
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SENSITIVE_TOKENS = ("key", "token", "secret", "password")


def _env_true(name: str, default: bool = False) -> bool:
    """Parse a truthy environment variable using common boolean spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort removal of secrets from run config."""
    def _is_sensitive(path: str) -> bool:
        key = path.lower()
        return any(tok in key for tok in SENSITIVE_TOKENS)

    def _redact(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                child = f"{path}.{k}" if path else str(k)
                if _is_sensitive(child):
                    out[k] = "***"
                else:
                    out[k] = _redact(v, child)
            return out
        if isinstance(value, list):
            return [_redact(v, f"{path}[{idx}]") for idx, v in enumerate(value)]
        return value

    return _redact(config)


def flatten_dict(data: Dict[str, Any], prefix: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten a nested dictionary into separator-delimited keys."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=kk, sep=sep))
        else:
            out[kk] = v
    return out


def add_wandb_args(parser) -> None:
    """Register shared CLI arguments used by scripts that support W&B logging."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (can also set WANDB_ENABLED=1)",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "fact"),
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "foundationmodels"),
        help="wandb entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP"),
        help="wandb group (optional)",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="wandb run name (optional)",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=[],
        help="wandb tags (space-separated)",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
        help="wandb mode",
    )


@dataclass
class WandbSession:
    """Fail-open wrapper around a W&B run object and artifact API."""
    _run: Any = None
    _wandb: Any = None

    @property
    def enabled(self) -> bool:
        """Return True when an active W&B run is available."""
        return self._run is not None

    def _disable(self) -> None:
        """Drop run references after an error to make future calls no-op safely."""
        self._run = None
        self._wandb = None

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics or payload dictionaries to W&B."""
        if not self.enabled:
            return
        try:
            if step is None:
                self._run.log(payload)
            else:
                self._run.log(payload, step=step)
        except Exception as exc:
            print(f"[wandb] log failed; disabling wandb session: {exc}", file=sys.stderr)
            self._disable()

    def update_summary(self, payload: Dict[str, Any]) -> None:
        """Update W&B run summary keys in a fail-open way."""
        if not self.enabled:
            return
        try:
            for k, v in payload.items():
                self._run.summary[k] = v
        except Exception as exc:
            print(f"[wandb] update_summary failed; disabling wandb session: {exc}", file=sys.stderr)
            self._disable()

    def _log_artifact(self, path: Path, *, name: str, artifact_type: str = "results") -> None:
        """Upload a file artifact when available and degrade safely on failure."""
        if not self.enabled or not path.exists():
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            print(f"[wandb] log_artifact failed; disabling wandb session: {exc}", file=sys.stderr)
            self._disable()

    def log_json_artifact(self, path: Path, *, name: str, artifact_type: str = "results") -> None:
        """Log a JSON artifact file for downstream inspection or reproducibility."""
        self._log_artifact(path, name=name, artifact_type=artifact_type)

    def log_file_artifact(self, path: Path, *, name: str, artifact_type: str = "results") -> None:
        """Log a generic file artifact with a custom artifact type."""
        self._log_artifact(path, name=name, artifact_type=artifact_type)

    def finish(self, *, exit_code: Optional[int] = None) -> None:
        """Finish the W&B run and always transition the session to disabled."""
        if not self.enabled:
            return
        try:
            if exit_code is not None:
                self._run.finish(exit_code=exit_code)
            else:
                self._run.finish()
        except Exception as exc:
            print(f"[wandb] finish failed; disabling wandb session: {exc}", file=sys.stderr)
        finally:
            self._disable()


def init_wandb_from_args(
    args: Any,
    *,
    run_name: str,
    job_type: str,
    config: Dict[str, Any],
    extra_tags: Optional[Iterable[str]] = None,
) -> WandbSession:
    """Create a fail-open W&B session from parsed CLI args and environment flags."""
    enabled = bool(getattr(args, "wandb", False)) or _env_true("WANDB_ENABLED", default=False)
    mode = str(getattr(args, "wandb_mode", "online"))
    if mode == "disabled":
        enabled = False
    if not enabled:
        return WandbSession()

    try:
        import wandb  # type: ignore

        tags: List[str] = list(getattr(args, "wandb_tags", []) or [])
        if extra_tags:
            tags.extend(list(extra_tags))

        safe_config = sanitize_config(config)
        run = wandb.init(
            project=getattr(args, "wandb_project", "fact"),
            entity=getattr(args, "wandb_entity", "foundationmodels"),
            group=getattr(args, "wandb_group", None),
            name=getattr(args, "wandb_name", None) or run_name,
            job_type=job_type,
            tags=tags or None,
            mode=mode,
            config=safe_config,
        )
        return WandbSession(_run=run, _wandb=wandb)
    except Exception as exc:
        print(f"[wandb] initialization failed; continuing without wandb logging: {exc}", file=sys.stderr)
        return WandbSession()


def extract_overall_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact metric dict from common payload schemas in this repo.

    This keeps script-specific logging simple and stable for dashboards.
    """
    summary: Dict[str, Any] = {}

    # Schema: {"results": [{"setup/classifier": "...", "metrics": {"overall": ...}}]}
    if isinstance(payload.get("results"), list):
        for row in payload["results"]:
            if not isinstance(row, dict):
                continue
            name = row.get("setup") or row.get("classifier")
            if not isinstance(name, str):
                continue
            overall = (row.get("metrics") or {}).get("overall")
            if isinstance(overall, dict):
                for k, v in overall.items():
                    summary[f"metrics/{name}/{k}"] = v
            if "evaluated_samples" in row:
                summary[f"metrics/{name}/evaluated_samples"] = row.get("evaluated_samples")
            if "failed_samples" in row:
                summary[f"metrics/{name}/failed_samples"] = row.get("failed_samples")

    # Schema: {"evaluation": {"leakage_percentages": {...}}}
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        leakage = evaluation.get("leakage_percentages")
        if isinstance(leakage, dict):
            for k, v in leakage.items():
                summary[f"leakage/{k}"] = v
        if "total_samples" in evaluation:
            summary["evaluation/total_samples"] = evaluation.get("total_samples")
        if "error_count" in evaluation:
            summary["evaluation/error_count"] = evaluation.get("error_count")

    # Schema: prevalence output
    prevalence = payload.get("prevalence")
    if isinstance(prevalence, dict):
        for k, v in prevalence.items():
            summary[f"prevalence/{k}"] = v

    counts = payload.get("counts")
    if isinstance(counts, dict) and "n" in counts:
        summary["counts/n"] = counts.get("n")

    return summary


def load_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))
