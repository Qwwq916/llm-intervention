import argparse
import json
from collections import Counter


def ngram_repetition_rate(text, n=5):
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
    parser.add_argument("--save_repetitive_file", type=str, default=None, help="If set, save samples that are determined to be duplicates.")
    args = parser.parse_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Unable to read or parse JSON file: {e}")
        exit(2)

    try:
        repetitive_samples = []
        rates = []

        for item in data:
            resp = item.get("response(intervention)", "")
            rrate = ngram_repetition_rate(resp, args.n)
            rates.append(rrate)
            if rrate > args.threshold:
                repetitive_samples.append(item)

        discrete_ratio = len(repetitive_samples) / len(data) if data else 0.0

        print(f"\n n-gram repetition rate: {discrete_ratio:.4f}")  # 保持原格式兼容输出

        if args.save_repetitive_file:
            with open(args.save_repetitive_file, "w", encoding="utf-8") as f:
                json.dump(repetitive_samples, f, ensure_ascii=False, indent=2)
            print(f"[Save] Repeated samples saved to: {args.save_repetitive_file}")

    except Exception as e:
        print(f"[Error] Failed to calculate the proportion of duplicate samples: {e}")
        exit(2)


if __name__ == "__main__":
    main()
