"""
Filter mt_train.jsonl to a subset of attack types for ablation experiments.

Usage:
    python scripts/filter_mt_dataset.py \
        --attacks priming self_correction \
        --input data/mt_train.jsonl \
        --output data/mt_train_priming_self_correction.jsonl

Presets: priming_only, self_correction_only, persona_only,
         priming_self_correction, all_three
"""
import json, argparse
from pathlib import Path

PRESETS = {
    "priming_only":            ["priming"],
    "self_correction_only":    ["self_correction"],
    "persona_only":            ["persona_switch"],
    "priming_self_correction": ["priming", "self_correction"],
    "all_three":               ["priming", "self_correction", "persona_switch"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", nargs="+",
                        help="Attack type names or a single preset key")
    parser.add_argument("--input", default="data/mt_train.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.attacks) == 1 and args.attacks[0] in PRESETS:
        allowed = set(PRESETS[args.attacks[0]])
    else:
        allowed = set(args.attacks)

    kept = []
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if row.get("attack_type") in allowed:
                kept.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    print(f"Filtered {len(kept)} conversations (attacks={allowed}) → {args.output}")


if __name__ == "__main__":
    main()
