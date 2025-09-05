import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# HarmBench-Llama-2-13b-cls classfier
model_name = "cais/HarmBench-Llama-2-13b-cls"

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

def ask_harmbench_judge(prompt, response):
    """Use the HarmBench classifier to determine whether the response is unsafe (jailbreak successful)."""
    pair = f"[PROMPT] {prompt}\n[RESPONSE] {response}"
    inputs = tokenizer(pair, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred == 1  # 1 表示 unsafe


def evaluate_nosafe_rate_harmbench(json_file_path):
    """Evaluate the nosafe rate (i.e., ASR) of a given layer of intervention output."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The input file content should be in list format.")

    total = len(data)
    unsafe_count = 0
    for item in tqdm(data, desc=f"Evaluating {os.path.basename(json_file_path)}"):
        prompt = item.get("prompt", "")
        response = item.get("response(intervention)", "")
        if ask_harmbench_judge(prompt, response):
            unsafe_count += 1

    return unsafe_count / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Use HarmBench to evaluate intervention outputs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    nosafe_rates = {}
    for filename in sorted(os.listdir(args.input_dir)):
        if filename.startswith("layer_") and filename.endswith("_intervention_outputs.json"):
            try:
                layer = int(filename.split("_")[1])
            except ValueError:
                continue
            json_path = os.path.join(args.input_dir, filename)
            rate = evaluate_nosafe_rate_harmbench(json_path)
            nosafe_rates[layer] = rate

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("Layer,NoSafeRate(HarmBench)\n")
        for layer in sorted(nosafe_rates.keys()):
            f.write(f"{layer},{nosafe_rates[layer]:.4f}\n")
            print(f"Layer {layer}: {nosafe_rates[layer]:.4f}")

    print(f"\nHarmBench nosafe rates saved to {args.output_file}")


if __name__ == "__main__":
    main()
