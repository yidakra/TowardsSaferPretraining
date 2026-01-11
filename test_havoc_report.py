#!/usr/bin/env python3
"""Test the HAVOC report generation logic."""

import json
from pathlib import Path
from tabulate import tabulate

# Table 10: HAVOC Leakage
print("\nTable 10: Model-Averaged Leakage on HAVOC (%)")
havoc_files = list(Path("results/havoc").glob("*_results.json"))
if havoc_files:
    # Load and aggregate results
    aggregated_results = {}
    total_models = 0

    for havoc_file in havoc_files:
        try:
            with open(havoc_file, 'r') as f:
                data = json.load(f)

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

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON in {havoc_file}: {e}")
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