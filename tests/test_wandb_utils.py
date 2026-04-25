"""Tests for W&B utility helpers."""

from src.utils.wandb import extract_overall_metrics, sanitize_config


def test_sanitize_config_redacts_nested_sensitive_keys():
    """Nested keys containing secret-like tokens should be redacted recursively."""
    cfg = {
        "api_key": "root-secret",
        "nested": {
            "token_value": "nested-secret",
            "safe": 1,
            "deep": {"password_hint": "should-hide", "ok": True},
        },
        "items": [
            {"secret_name": "x"},
            {"value": "visible"},
        ],
    }

    got = sanitize_config(cfg)

    assert got["api_key"] == "***"
    assert got["nested"]["token_value"] == "***"
    assert got["nested"]["safe"] == 1
    assert got["nested"]["deep"]["password_hint"] == "***"
    assert got["nested"]["deep"]["ok"] is True
    assert got["items"][0]["secret_name"] == "***"
    assert got["items"][1]["value"] == "visible"


def test_extract_overall_metrics_covers_supported_payload_shapes():
    """Metric extraction should include all documented payload schema branches."""
    payload = {
        "results": [
            {
                "setup": "harmformer",
                "metrics": {"overall": {"precision": 0.8, "f1": 0.7}},
                "evaluated_samples": 10,
                "failed_samples": 1,
            }
        ],
        "evaluation": {
            "leakage_percentages": {"neutral": 12.5},
            "total_samples": 40,
            "error_count": 2,
        },
        "prevalence": {"overall_toxic": 0.4},
        "counts": {"n": 100},
    }

    got = extract_overall_metrics(payload)

    assert got["metrics/harmformer/precision"] == 0.8
    assert got["metrics/harmformer/f1"] == 0.7
    assert got["metrics/harmformer/evaluated_samples"] == 10
    assert got["metrics/harmformer/failed_samples"] == 1
    assert got["leakage/neutral"] == 12.5
    assert got["evaluation/total_samples"] == 40
    assert got["evaluation/error_count"] == 2
    assert got["prevalence/overall_toxic"] == 0.4
    assert got["counts/n"] == 100
