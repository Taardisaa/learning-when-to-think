"""Generate synthetic SFT training data from GSM8K.

Usage:
    python scripts/generate_sft_data.py
    python scripts/generate_sft_data.py --subset 500 --output data/sft_small.jsonl
"""

import argparse

from src.data.synthetic import generate_sft_dataset, save_sft_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data")
    parser.add_argument(
        "--output", default="data/sft_train.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Limit number of GSM8K problems"
    )
    parser.add_argument(
        "--backoff-ratio", type=float, default=0.4, help="Fraction of backoff examples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating SFT data (backoff_ratio={args.backoff_ratio})...")
    examples = generate_sft_dataset(
        split="train",
        subset_size=args.subset,
        backoff_ratio=args.backoff_ratio,
        seed=args.seed,
    )

    backoff_count = sum(1 for ex in examples if ex["has_backoff"])
    print(f"Total examples: {len(examples)}")
    print(f"  Clean:   {len(examples) - backoff_count}")
    print(f"  Backoff: {backoff_count} ({100 * backoff_count / len(examples):.1f}%)")

    save_sft_dataset(examples, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
