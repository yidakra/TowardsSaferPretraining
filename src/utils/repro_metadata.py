"""Utilities for capturing run metadata for reproducibility.

This repo runs in mixed environments (local, Snellius/Slurm, API-backed). Small
differences (package versions, model ids, dates) can materially affect results,
so we persist lightweight metadata alongside every JSON artifact.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL, timeout=2)
        return out.decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def _file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _pkg_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    try:
        from importlib import metadata as importlib_metadata  # py3.8+
    except Exception:  # pragma: no cover
        importlib_metadata = None  # type: ignore

    for name in packages:
        v: Optional[str] = None
        if importlib_metadata is not None:
            try:
                v = importlib_metadata.version(name)
            except Exception:
                v = None
        versions[name] = v
    return versions


def gather_run_metadata(*, repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Gather best-effort metadata about the current run.

    This must not fail the evaluation pipeline.
    """

    root = Path(repo_root or os.getcwd())

    git_commit = _safe_run(["git", "rev-parse", "HEAD"], cwd=root)
    git_is_dirty = _safe_run(["git", "status", "--porcelain"], cwd=root)

    requirements_path = root / "requirements.txt"
    requirements_sha = _file_sha256(requirements_path) if requirements_path.exists() else None

    return {
        "timestamp_utc": _iso_now(),
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "git": {
            "commit": git_commit,
            "dirty": bool(git_is_dirty),
        },
        "inputs": {
            "requirements_txt": str(requirements_path) if requirements_path.exists() else None,
            "requirements_sha256": requirements_sha,
        },
        "packages": _pkg_versions(
            [
                "torch",
                "transformers",
                "openai",
                "datasets",
                "accelerate",
                "scikit-learn",
                "codecarbon",
                "google-genai",
                "google-generativeai",
            ]
        ),
    }
