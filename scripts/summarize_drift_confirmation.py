#!/usr/bin/env python3
"""Summarise the pinned-snapshot drift-confirmation runs into paste-ready numbers.

Consumes the result JSONs produced by `evaluate_ttp_eval.py` (pinned + floating
`gpt-4o` setups, seeded, with system_fingerprint capture) and by
`ttp_noise_floor.py`, and prints:

  1. a per-setup table (model alias, evaluated n, P/R/F1, seed, the distinct
     system_fingerprint values seen) — the direct snapshot evidence for
     reviewer item 1;
  2. the same-snapshot noise floor (F1 range across identical repeats), so the
     April->May swing can be quoted against it;
  3. the --invalid-policy sensitivity for the headline row (exclude / non_toxic
     / toxic) — reviewer item 4.

Reads only local JSON; no API spend. Point it at the results directory:

    python scripts/summarize_drift_confirmation.py results/ttp_eval_drift_confirmation
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _fmt_fp(stats: Dict[str, Any] | None) -> str:
    if not stats:
        return "(no client_stats)"
    fps = stats.get("system_fingerprints") or {}
    if not fps:
        return "(none logged)"
    # {"fp_abc": 393} -> "fp_abc x393"
    return ", ".join(f"{(k or '∅')} x{v}" for k, v in fps.items())


def _iter_setups(payload: Dict[str, Any]):
    """Yield (setup_name, evaluated, metrics, client_stats, seed) per setup."""
    seed = (payload.get("evaluation_config") or {}).get("seed")
    for r in payload.get("results", []):
        m = (r.get("metrics") or {}).get("overall") or {}
        yield (
            r.get("setup", "?"),
            r.get("evaluated_samples"),
            m.get("precision"), m.get("recall"), m.get("f1"),
            r.get("invalid_policy"),
            r.get("client_stats"),
            seed,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", help="Directory containing the drift-confirmation *.json outputs")
    p.add_argument("--noise-floor-glob", default="*noise_floor*.json")
    args = p.parse_args()

    rd = Path(args.results_dir)
    if not rd.exists():
        raise SystemExit(f"No such directory: {rd}")

    snapshot_files = sorted(f for f in glob.glob(str(rd / "*.json"))
                            if "noise_floor" not in os.path.basename(f))
    noise_files = sorted(glob.glob(str(rd / args.noise_floor_glob)))

    print("=" * 78)
    print("1. PER-SNAPSHOT F1 + system_fingerprint  (reviewer item 1)")
    print("=" * 78)
    header = f"{'setup':<40}{'n':>5}{'P':>7}{'R':>7}{'F1':>7}  fingerprint(s)"
    print(header)
    print("-" * 78)
    # Collect the headline row across invalid policies for section 3.
    sensitivity: List[tuple] = []
    for f in snapshot_files:
        payload = _load(f)
        for name, n, pr, rc, f1, policy, stats, seed in _iter_setups(payload):
            def s(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
            # The setup name is the same for every leg; the file stem carries
            # which snapshot/policy this row is (floating_…, pinned_…, sensitivity_…).
            row_label = Path(f).stem
            print(f"{row_label[:40]:<40}{(n if n is not None else '-'): >5}{s(pr):>7}{s(rc):>7}{s(f1):>7}  {_fmt_fp(stats)}")
            sensitivity.append((os.path.basename(f), name, policy, pr, rc, f1))
    if not snapshot_files:
        print("(no snapshot result JSONs found)")

    print()
    print("=" * 78)
    print("2. SAME-SNAPSHOT NOISE FLOOR  (reviewer item 1 corroboration)")
    print("=" * 78)
    if noise_files:
        for f in noise_files:
            nf = _load(f).get("noise_floor", {})
            fps = nf.get("system_fingerprints") or {}
            print(f"{os.path.basename(f)}")
            print(f"  setup            : {nf.get('setup')}")
            print(f"  repeats          : {nf.get('repeats')}   seed={nf.get('seed')}")
            print(f"  F1 per run       : {[round(x,4) for x in nf.get('f1_per_run', [])]}")
            print(f"  F1 range (noise) : {nf.get('f1_range')}")
            print(f"  unstable samples : {nf.get('n_unstable_samples')}/{nf.get('n_samples')}")
            print(f"  fingerprints     : {', '.join(f'{k or chr(8709)} x{v}' for k,v in fps.items()) or '(none)'}")
    else:
        print("(no noise-floor JSON found — run scripts/ttp_noise_floor.py)")

    print()
    print("=" * 78)
    print("3. --invalid-policy SENSITIVITY, headline row  (reviewer item 4)")
    print("=" * 78)
    # Only the sensitivity_* files belong here — the pinned-snapshot legs also
    # run under `exclude` and would otherwise pollute the policy comparison.
    by_policy = {}
    for fname, name, policy, pr, rc, f1 in sensitivity:
        if policy and fname.startswith("sensitivity_"):
            by_policy.setdefault(policy, []).append((name, pr, rc, f1, fname))
    if len(by_policy) >= 2:
        print(f"{'policy':<12}{'setup':<38}{'P':>7}{'R':>7}{'F1':>7}")
        print("-" * 72)
        for policy in ("exclude", "non_toxic", "toxic"):
            for name, pr, rc, f1, _ in by_policy.get(policy, []):
                def s(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
                print(f"{policy:<12}{name[:38]:<38}{s(pr):>7}{s(rc):>7}{s(f1):>7}")
    else:
        print("(need runs under >=2 --invalid-policy values; see jobs/run_drift_confirmation.sh)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
