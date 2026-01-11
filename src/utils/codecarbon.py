"""
Optional CodeCarbon integration.

Enabled by default (fail-open).

Disable by setting environment variable:
  DISABLE_CODECARBON=1

Outputs will be written to CODECARBON_OUTPUT_DIR (default: results/codecarbon).
"""

from __future__ import annotations

import os
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s).strip("_")


@contextmanager
def maybe_track_emissions(run_name: str) -> Iterator[None]:
    """
    Context manager that tracks emissions via CodeCarbon when enabled.

    This is designed to be safe in environments where CodeCarbon is not installed
    or where it cannot initialize (it will silently no-op).
    """
    if os.environ.get("DISABLE_CODECARBON") == "1":
        yield
        return

    try:
        from codecarbon import EmissionsTracker  # type: ignore
    except Exception:
        yield
        return

    out_dir = Path(os.environ.get("CODECARBON_OUTPUT_DIR", "results/codecarbon"))
    out_dir.mkdir(parents=True, exist_ok=True)

    project_name = os.environ.get("CODECARBON_PROJECT_NAME", "TowardsSaferPretraining")
    experiment_id = os.environ.get("CODECARBON_EXPERIMENT_ID")  # optional

    safe_run = _safe_filename(run_name) or "run"
    safe_exp = _safe_filename(experiment_id) if experiment_id else ""
    output_file = f"emissions_{safe_run}{'_' + safe_exp if safe_exp else ''}.csv"

    # Build kwargs carefully: CodeCarbon's constructor signature changes across versions.
    tracker_kwargs = {
        "project_name": project_name,
        "experiment_id": experiment_id,
        "output_dir": str(out_dir),
        "output_file": output_file,
        "measure_power_secs": 1,
        "log_level": os.environ.get("CODECARBON_LOG_LEVEL", "error"),
    }

    # Optional: pin location if supported by this CodeCarbon version
    country_iso_code: Optional[str] = os.environ.get("CODECARBON_COUNTRY_ISO_CODE")
    if country_iso_code:
        tracker_kwargs["country_iso_code"] = country_iso_code

    try:
        tracker = EmissionsTracker(**tracker_kwargs)
    except TypeError as e:
        # Common incompatibility: older/newer CodeCarbon versions may not accept some kwargs.
        msg = str(e)
        if "country_iso_code" in msg and "unexpected keyword" in msg:
            tracker_kwargs.pop("country_iso_code", None)
            tracker = EmissionsTracker(**tracker_kwargs)
        else:
            raise

    try:
        tracker.start()
    except Exception:
        # fail-open: don't block experiments
        yield
        return

    try:
        yield
    finally:
        try:
            # CodeCarbon writes one row per tracker stop.
            tracker.stop()
        except Exception:
            pass

        # Also write a JSON sidecar for consistency with the rest of the repo's results.
        # We keep CSV for easy aggregation, but JSON is nicer for programmatic consumption.
        try:
            csv_path = out_dir / output_file
            json_path = out_dir / output_file.replace(".csv", ".json")

            # Some CodeCarbon versions flush the CSV slightly after stop() returns.
            rows = []
            for _ in range(25):  # around 5s max
                if csv_path.exists():
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        break
                import time
                time.sleep(0.2)

            if rows:
                payload = rows[-1]
                payload["_run_name"] = run_name
                payload["_experiment_id"] = experiment_id
                payload["_csv_path"] = str(csv_path)
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(payload, jf, indent=2)
        except Exception:
            # fail-open: don't block experiments
            pass
