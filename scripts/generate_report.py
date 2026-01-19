#!/usr/bin/env python3
"""Generate reproduction report matching paper tables."""

import json
from pathlib import Path
from tabulate import tabulate  # type: ignore

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as e:
        raise RuntimeError(f"I/O error accessing JSON file: {path} ({e})") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Malformed JSON in file: {path} ({e})") from e

print("=" * 80)
print("REPRODUCTION REPORT - Mendu et al. 2025")
print("=" * 80)

# Table 3: TTP Quality on TTP-Eval
print("\nTable 3: TTP Quality (Toxic Dimension)")
try:
    # Preferred output path
    try:
        ttp_payload = load_json("results/ttp_eval/results.json")
    except RuntimeError:
        # Backwards/alternative job output name
        ttp_payload = load_json("results/ttp_eval/ttp_results.json")
    # Unified schema: find the OpenAI TTP result entry
    ttp_entry = None
    for r in ttp_payload.get("results", []):
        setup = (r.get("setup") or "").lower()
        if setup.startswith("ttp ("):
            ttp_entry = r
            break
    if ttp_entry is None:
        raise RuntimeError("No TTP entry found in results/ttp_eval/*.json")

    metrics = ttp_entry.get("metrics", {}).get("per_harm", {})
    overall = ttp_entry.get("metrics", {}).get("overall", {})

    table3_data = [
        ["Hate & Violence",
         metrics.get("H", {}).get("precision", "N/A"),
         metrics.get("H", {}).get("recall", "N/A"),
         metrics.get("H", {}).get("f1", "N/A")],
        ["Ideological Harm",
         metrics.get("IH", {}).get("precision", "N/A"),
         metrics.get("IH", {}).get("recall", "N/A"),
         metrics.get("IH", {}).get("f1", "N/A")],
        ["Sexual",
         metrics.get("SE", {}).get("precision", "N/A"),
         metrics.get("SE", {}).get("recall", "N/A"),
         metrics.get("SE", {}).get("f1", "N/A")],
        ["Illegal",
         metrics.get("IL", {}).get("precision", "N/A"),
         metrics.get("IL", {}).get("recall", "N/A"),
         metrics.get("IL", {}).get("f1", "N/A")],
        ["Self-Inflicted",
         metrics.get("SI", {}).get("precision", "N/A"),
         metrics.get("SI", {}).get("recall", "N/A"),
         metrics.get("SI", {}).get("f1", "N/A")],
        ["Toxic (Overall)",
         overall.get("precision", "N/A"),
         overall.get("recall", "N/A"),
         overall.get("f1", "N/A")],
    ]
except RuntimeError as e:
    print(f"Warning: Could not load TTP results ({e}).")
    table3_data = [
        ["Hate & Violence", "N/A", "N/A", "N/A"],
        ["Ideological Harm", "N/A", "N/A", "N/A"],
        ["Sexual", "N/A", "N/A", "N/A"],
        ["Illegal", "N/A", "N/A", "N/A"],
        ["Self-Inflicted", "N/A", "N/A", "N/A"],
        ["Toxic (Overall)", "N/A", "N/A", "N/A"],
    ]
print(tabulate(table3_data, headers=["Harm", "Precision", "Recall", "F1"], tablefmt="grid"))

# Table 4: TTP vs Perspective on TTP-Eval (Toxic dimension)
print("\nTable 4: TTP vs Perspective (Toxic Dimension, TTP-Eval)")
try:
    try:
        table4 = load_json("results/ttp_eval_baselines/results.json")
    except RuntimeError:
        # PR job scripts may split outputs.
        # Combine rows from any present files.
        combined_results = []
        for p in [
            "results/ttp_eval_baselines/table4_perspective_openai_ttp.json",
            "results/ttp_eval_baselines/table4_local_ttp.json",
        ]:
            try:
                payload = load_json(p)
                combined_results.extend(payload.get("results", []))
            except RuntimeError:
                pass
        if not combined_results:
            raise RuntimeError("No results found in results/ttp_eval_baselines/*")
        table4 = {"results": combined_results}
    rows = []
    for r in table4.get("results", []):
        m = r.get("metrics", {}).get("overall", {})
        rows.append([r.get("setup", "Unknown"), m.get("precision", "N/A"), m.get("recall", "N/A"), m.get("f1", "N/A")])
    if not rows:
        raise RuntimeError("No results found in results/ttp_eval_baselines/*.json")
    print(tabulate(rows, headers=["Setup", "Precision", "Recall", "F1"], tablefmt="grid"))
except RuntimeError as e:
    print(f"Warning: Could not load Table 4 results ({e}).")

# Table 6: HarmFormer Quality
print("\nTable 6: HarmFormer Quality (Toxic Dimension)")
print("Note: The paper's Table 6 is reported on the authors' internal HarmFormer test split, which is not publicly released.")
print("This repo reports HarmFormer performance on TTP-Eval as a proxy.")
try:
    harmformer_results = load_json("results/harmformer/harmformer_results.json")
    metrics = harmformer_results.get("metrics", {}).get("per_harm", {})
    overall = harmformer_results.get("metrics", {}).get("overall", {})

    table6_data = [
        ["Hate & Violence", metrics.get("H", {}).get("precision", "N/A"), metrics.get("H", {}).get("recall", "N/A"), metrics.get("H", {}).get("f1", "N/A")],
        ["Ideological Harm", metrics.get("IH", {}).get("precision", "N/A"), metrics.get("IH", {}).get("recall", "N/A"), metrics.get("IH", {}).get("f1", "N/A")],
        ["Sexual", metrics.get("SE", {}).get("precision", "N/A"), metrics.get("SE", {}).get("recall", "N/A"), metrics.get("SE", {}).get("f1", "N/A")],
        ["Illegal", metrics.get("IL", {}).get("precision", "N/A"), metrics.get("IL", {}).get("recall", "N/A"), metrics.get("IL", {}).get("f1", "N/A")],
        ["Self-Inflicted", metrics.get("SI", {}).get("precision", "N/A"), metrics.get("SI", {}).get("recall", "N/A"), metrics.get("SI", {}).get("f1", "N/A")],
        ["Toxic (Overall)", overall.get("precision", "N/A"), overall.get("recall", "N/A"), overall.get("f1", "N/A")],
    ]
except RuntimeError as e:
    print(f"Warning: Could not load HarmFormer results ({e}).")
    table6_data = [
        ["Hate & Violence", "N/A", "N/A", "N/A"],
        ["Ideological Harm", "N/A", "N/A", "N/A"],
        ["Sexual", "N/A", "N/A", "N/A"],
        ["Illegal", "N/A", "N/A", "N/A"],
        ["Self-Inflicted", "N/A", "N/A", "N/A"],
        ["Toxic (Overall)", "N/A", "N/A", "N/A"],
    ]
print(tabulate(table6_data, headers=["Harm", "Precision", "Recall", "F1"], tablefmt="grid"))

# Table 7: OpenAI Moderation test set
print("\nTable 7: Performance on OpenAI Moderation Dataset (Binary Toxic Label)")
try:
    try:
        mod_results = load_json("results/moderation/table7_results.json")
    except RuntimeError:
        # PR job scripts may split API vs local runs.
        combined = []
        for p in [
            "results/moderation/table7_api_results.json",
            "results/moderation/table7_local_results.json",
        ]:
            try:
                payload = load_json(p)
                combined.extend(payload.get("results", []))
            except RuntimeError:
                pass
        if not combined:
            raise RuntimeError("No results found in results/moderation/table7_*.json")
        mod_results = {"results": combined}
    rows = []
    for r in mod_results.get("results", []):
        m = r.get("metrics", {}).get("overall", {})
        rows.append([r.get("classifier", "Unknown"), m.get("precision", "N/A"), m.get("recall", "N/A"), m.get("f1", "N/A")])
    if not rows:
        raise RuntimeError("No results found in results/moderation/table7_*.json")
    print(tabulate(rows, headers=["Setup", "Precision", "Recall", "F1"], tablefmt="grid"))
except RuntimeError as e:
    print(f"Warning: Could not load Table 7 results ({e}).")

# Table 10: HAVOC Leakage
print("\nTable 10: Model-Averaged Leakage on HAVOC (%)")
havoc_files = list(Path("results/havoc").glob("*_results.json"))
if havoc_files:
    # Load and aggregate results
    aggregated_results = {}
    total_models = 0

    for havoc_file in havoc_files:
        try:
            data = load_json(havoc_file)

            # Validate expected structure
            if "evaluation" not in data:
                print(f"Warning: Missing 'evaluation' key in {havoc_file}")
                continue

            evaluation = data["evaluation"]

            # Extract model name from filename (remove '_results.json' suffix)
            model_name = havoc_file.stem.replace('_results', '')

            # Validate expected keys
            if "leakage_percentages" not in evaluation:
                print(f"Warning: Missing 'leakage_percentages' in {havoc_file}")
                continue

            leakage_pct = evaluation["leakage_percentages"]

            # Accumulate results for this model
            aggregated_results[model_name] = {
                "neutral": leakage_pct.get("neutral", 0.0),
                "passive": leakage_pct.get("passive", 0.0),
                "provocative": leakage_pct.get("provocative", 0.0),
                "overall": leakage_pct.get("overall", 0.0),
            }
            total_models += 1

        except RuntimeError as e:
            print(f"Warning: Could not load HAVOC results from {havoc_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {havoc_file}: {e}")

    if aggregated_results:
        # Prepare table data
        table_data = []
        overall_totals = {"neutral": 0.0, "passive": 0.0, "provocative": 0.0, "overall": 0.0}

        for model_name, metrics in aggregated_results.items():
            table_data.append([
                model_name,
                f"{metrics['neutral']:.2f}",
                f"{metrics['passive']:.2f}",
                f"{metrics['provocative']:.2f}",
                f"{metrics['overall']:.2f}"
            ])

            # Accumulate for overall average
            for key in overall_totals:
                overall_totals[key] += metrics[key]

        # Add overall average row
        if total_models > 0:
            table_data.append([
                "Average",
                f"{overall_totals['neutral']/total_models:.2f}",
                f"{overall_totals['passive']/total_models:.2f}",
                f"{overall_totals['provocative']/total_models:.2f}",
                f"{overall_totals['overall']/total_models:.2f}"
            ])

        print(tabulate(table_data, headers=["Model", "Neutral", "Passive", "Provocative", "Overall"], tablefmt="grid"))
    else:
        print("No valid HAVOC results found.")
else:
    print("No HAVOC results found. Run Phase 2 first.")
