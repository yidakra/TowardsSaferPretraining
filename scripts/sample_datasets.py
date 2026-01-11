#!/usr/bin/env python3
"""Sample datasets for analysis (100K samples each to stay in budget)."""

from datasets import load_dataset  # type: ignore
import json
from pathlib import Path
from tqdm import tqdm  # type: ignore

def sample_c4(n_samples=100000):
    """Sample from C4 dataset."""
    print(f"Sampling {n_samples} from C4...")

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []

    for i, item in enumerate(tqdm(dataset, total=n_samples)):
        if i >= n_samples:
            break
        samples.append({
            "url": item.get("url", f"c4_sample_{i}"),
            "text": item["text"][:4096]  # Truncate to 4K chars
        })

    # Save
    output_file = Path("datasets/samples/c4_100k.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved to {output_file}")

def sample_fineweb(n_samples=100000):
    """Sample from FineWeb dataset."""
    print(f"Sampling {n_samples} from FineWeb...")

    dataset = load_dataset("HuggingFaceFW/fineweb",
                          name="CC-MAIN-2024-10",  # Use one dump
                          split="train",
                          streaming=True)
    samples = []

    for i, item in enumerate(tqdm(dataset, total=n_samples)):
        if i >= n_samples:
            break
        samples.append({
            "url": item.get("url", f"fineweb_sample_{i}"),
            "text": item["text"][:4096]
        })

    output_file = Path("datasets/samples/fineweb_100k.jsonl")
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    sample_c4(100000)
    sample_fineweb(100000)
    print("\nDataset sampling complete!")
