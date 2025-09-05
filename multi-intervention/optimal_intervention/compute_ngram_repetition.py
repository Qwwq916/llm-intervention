
import argparse
import json
from collections import Counter


def ngram_repetition_rate(text, n=8):
    tokens = text.strip().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counter = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(count for count in counter.values() if count > 1)
    return repeated / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Proportion of duplicate samples in the judgment formula")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for determining duplicate output")
    args = parser.parse_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Unable to read or parse JSON file: {e}")
        exit(2)

    try:
        responses = [item.get("response(intervention)", "") for item in data]
        rates = [ngram_repetition_rate(resp, args.n) for resp in responses]
        repetitive_count = sum(1 for r in rates if r > args.threshold)
        discrete_ratio = repetitive_count / len(rates) if rates else 0.0

        print(f"\n n-gram repeat rate: {discrete_ratio:.4f}")  # Maintain an output format compatible with the original script.
    except Exception as e:
        print(f"[Error] Failed to calculate the proportion of duplicate samples: {e}")
        exit(2)

    return discrete_ratio
if __name__ == "__main__":
    main()
