#!/usr/bin/env python3
"""Aggregate HAVOC results across all models."""

import json
from pathlib import Path

results_dir = Path("results/havoc")
all_results = []

for result_file in results_dir.glob("*_results.json"):
    with open(result_file) as f:
        data = json.load(f)
        all_results.append({
            "model": result_file.stem.replace("_results", ""),
            "metrics": data
        })

# Calculate aggregated metrics
print("\n=== HAVOC Results Summary ===\n")
for result in all_results:
    print(f"Model: {result['model']}")
    metrics = result['metrics']
    # Print key metrics matching Table 10
    print(f"  Neutral Leak: {metrics.get('neutral_leak', 'N/A'):.2f}%")
    print(f"  Passive Leak: {metrics.get('passive_leak', 'N/A'):.2f}%")
    print(f"  Provocative Leak: {metrics.get('provocative_leak', 'N/A'):.2f}%")
    print(f"  Overall Leak: {metrics.get('overall_leak', 'N/A'):.2f}%")
    print()

# Save aggregated results
with open("results/havoc/aggregated_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Aggregated results saved to: results/havoc/aggregated_results.json")
