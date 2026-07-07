#!/usr/bin/env python3
"""Recompute the --invalid-policy sensitivity offline from ONE exclude-policy run.

Reviewer item 4 asks how the headline TTP row moves under the three invalid-
output policies (exclude / non_toxic / toxic). Rerunning the benchmark three
times costs ~$8 per pass — but the three policies differ ONLY in how API-failed
samples are counted, so given one `exclude` run we can recompute the other two
exactly, for free:

  * The run JSON stores per-sample (pred, gold) for every *evaluated* sample,
    from which TP/FP/FN follow directly.
  * The failed samples' gold split is recoverable without knowing which rows
    failed: (# gold-toxic among failed) = (total gold-toxic in the TSV) -
    (# gold-toxic among evaluated).
  * non_toxic policy: every failed sample becomes a non-toxic prediction ->
    failed gold-toxic samples are extra FNs.
  * toxic policy: every failed sample becomes a toxic prediction -> failed
    gold-toxic are extra TPs, failed gold-non-toxic extra FPs.

Writes summarizer-compatible sensitivity_{exclude,non_toxic,toxic}.json files
next to the input (or --out-dir) and prints the table.

    python scripts/recompute_invalid_sensitivity.py \
        results/ttp_eval_drift_confirmation/floating_gpt-4o.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL_COLS = ["Hate&V Lab", "Ideologi Lab", "Sexual Lab", "Illegal Lab", "Self-Infl Lab"]


def _total_gold_toxic(tsv_path: Path) -> int:
    """Count gold-toxic rows with the project's own label parser.

    Must match the evaluation loader exactly (e.g. one TSV row uses the S0/S1/S2
    encoding, which a naive `== "intent"` comparison silently misses).
    """
    from src.utils.taxonomy import Dimension
    csv.field_size_limit(10_000_000)
    with tsv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return sum(
        1 for r in rows
        if any(Dimension.from_label(r.get(c) or "") == Dimension.TOXIC for c in LABEL_COLS)
    ), len(rows)


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_json", help="Result JSON of an --invalid-policy exclude run")
    ap.add_argument("--data-path", default=str(ROOT / "data" / "TTP-Eval" / "TTPEval.tsv"))
    ap.add_argument("--out-dir", help="Where to write sensitivity_*.json (default: alongside run_json)")
    args = ap.parse_args()

    run_path = Path(args.run_json)
    payload = json.loads(run_path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else run_path.parent

    total_toxic, total_rows = _total_gold_toxic(Path(args.data_path))

    for result in payload.get("results", []):
        if result.get("invalid_policy") != "exclude":
            raise SystemExit(
                f"Run has invalid_policy={result.get('invalid_policy')!r}; "
                "offline recomputation needs an `exclude` run as the base."
            )
        ps = result.get("per_sample_toxic") or {}
        preds, golds = ps.get("pred"), ps.get("gold")
        if not preds or not golds or len(preds) != len(golds):
            raise SystemExit("Run JSON lacks usable per_sample_toxic data.")
        failed = result.get("failed_samples") or 0
        if len(preds) + failed != total_rows:
            raise SystemExit(
                f"Sample accounting mismatch: {len(preds)} evaluated + {failed} failed "
                f"!= {total_rows} TSV rows (was the run made with --limit?)"
            )

        tp = sum(1 for p, g in zip(preds, golds) if p and g)
        fp = sum(1 for p, g in zip(preds, golds) if p and not g)
        fn = sum(1 for p, g in zip(preds, golds) if not p and g)
        evaluated_toxic = sum(1 for g in golds if g)
        failed_toxic = total_toxic - evaluated_toxic       # gold-toxic among failed
        failed_safe = failed - failed_toxic                 # gold-non-toxic among failed
        assert 0 <= failed_toxic <= failed, "failed gold split out of range"

        variants = {
            "exclude": _prf(tp, fp, fn),
            "non_toxic": _prf(tp, fp, fn + failed_toxic),
            "toxic": _prf(tp + failed_toxic, fp + failed_safe, fn),
        }

        # Sanity: our exclude recomputation must reproduce the run's own stored
        # metrics; a mismatch means the metric definitions have drifted.
        stored = (result.get("metrics") or {}).get("overall") or {}
        for k in ("precision", "recall", "f1"):
            if isinstance(stored.get(k), (int, float)) and abs(stored[k] - variants["exclude"][k]) > 5e-4:
                raise SystemExit(
                    f"Recomputed exclude {k}={variants['exclude'][k]:.4f} disagrees with "
                    f"stored {k}={stored[k]:.4f} — refusing to emit sensitivity numbers."
                )

        name = result.get("setup", "?")
        print(f"setup={name}  evaluated={len(preds)}  failed={failed} "
              f"(gold-toxic among failed: {failed_toxic})")
        print(f"{'policy':<12}{'P':>8}{'R':>8}{'F1':>8}")
        for pol, m in variants.items():
            print(f"{pol:<12}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
        span = max(m["f1"] for m in variants.values()) - min(m["f1"] for m in variants.values())
        print(f"F1 span across policies: {span:.3f}"
              + ("  (no failures -> policy is moot)" if failed == 0 else ""))

        # Emit summarizer-compatible files so summarize_drift_confirmation.py
        # renders section 3 without any extra API passes.
        for pol, m in variants.items():
            n_eval = len(preds) if pol == "exclude" else len(preds) + failed
            out = {
                "evaluation_config": payload.get("evaluation_config"),
                "recomputed_offline_from": run_path.name,
                "results": [{
                    "setup": name,
                    "total_samples": result.get("total_samples"),
                    "failed_samples": failed,
                    "evaluated_samples": n_eval,
                    "invalid_policy": pol,
                    "metrics": {"overall": m},
                    "client_stats": result.get("client_stats"),
                }],
            }
            out_path = out_dir / f"sensitivity_{pol}.json"
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
