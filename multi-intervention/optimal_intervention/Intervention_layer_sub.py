
import json
import argparse
import os
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_dataset, load_model_and_tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--anchors", type=str, nargs="+", default=["./data/benigntest.txt", "./data/harmfultest.txt"])
    parser.add_argument("--datasets", type=str, nargs="+", default="./results/harmful.csv")
    parser.add_argument("--text_columns", type=str, nargs="+")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--system_prompt", type=str)
    parser.add_argument("--separation_file", type=str, required=True)
    parser.add_argument("--intervention_layers", type=int, nargs="+", default=[14, 15])
    parser.add_argument("--intervention_strengths", type=float, nargs="+", default=[0.4, 0.7])
    parser.add_argument("--precomputed_directions", type=str, default=None)
    parser.add_argument("--save_directions", type=str, default=None)
    return parser.parse_args()


def apply_chat_template(tokenizer, texts, system_prompt=None):
    full_prompt_list = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt_list.append(full_prompt)
    return full_prompt_list


def get_hidden_states(model, tokenizer, full_prompt_list, start_layer, end_layer):
    model.eval()
    all_layer_hidden_states = [[] for _ in range(end_layer - start_layer + 1)]
    with torch.no_grad():
        for full_prompt in tqdm(full_prompt_list, desc="Calculating hidden states"):
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            for layer in range(start_layer, end_layer + 1):
                all_layer_hidden_states[layer - start_layer].append(outputs.hidden_states[layer][:, -1, :])
    for i in range(len(all_layer_hidden_states)):
        all_layer_hidden_states[i] = torch.stack(all_layer_hidden_states[i])
    return all_layer_hidden_states


def compute_difference_vectors(harmless_embeddings, harmful_embeddings):
    return torch.stack([harmful - harmless for harmless, harmful in zip(harmless_embeddings, harmful_embeddings)])


def register_hooks(model, intervention_layers, direction_vectors, strength_dict):
    hooks = []
    def hook(module, input, output, layer_idx):
        direction = direction_vectors.get(layer_idx)
        strength = strength_dict.get(layer_idx, 1.0)
        if direction is not None:
            output[0][:, -1, :] -= strength * direction.squeeze(0)
        return output
    for i, layer in enumerate(model.model.layers):
        if i in intervention_layers:
            hook_fn = lambda module, input, output, layer_idx=i: hook(module, input, output, layer_idx)
            hook_handle = layer.register_forward_hook(hook_fn)
            hooks.append(hook_handle)
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def generate_response_with_intervention(model, tokenizer, prompt, direction_vectors, intervention_layers,
                                         strength_dict, max_new_tokens=256):
    model.eval()
    with torch.no_grad():
        full_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        hooks = register_hooks(model, intervention_layers, direction_vectors, strength_dict)
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        remove_hooks(hooks)
        return generated_text


def evaluate_and_save_outputs(model, tokenizer, prompts, direction_vectors, intervention_layers, save_path, strength_dict):
    results = []
    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        intv_text = generate_response_with_intervention(model, tokenizer, prompt, direction_vectors, intervention_layers, strength_dict)
        results.append({
            "prompt": prompt,
            "response(intervention)": intv_text.strip()
        })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"save_path {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if len(args.intervention_layers) != len(args.intervention_strengths):
        raise ValueError("The length of intervention_layers and intervention_strengths must be consistent")
    strength_dict = {layer: strength for layer, strength in zip(args.intervention_layers, args.intervention_strengths)}

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.precomputed_directions and os.path.exists(args.precomputed_directions):
        with open(args.precomputed_directions, "r") as f:
            all_vectors = json.load(f)
            direction_vectors = {
            int(k): torch.tensor(v).to(model.device)
            for k, v in all_vectors.items()
            if int(k) in args.intervention_layers                                             }
            # direction_vectors = {int(k): torch.tensor(v).to(model.device) for k, v in json.load(f).items()}
        print(f"[Cache loading] Precomputed difference vectors loaded: {list(direction_vectors.keys())}")
    else:
        df_anchors_list = [load_dataset(p) for p in args.anchors]
        full_prompts_0 = apply_chat_template(tokenizer, df_anchors_list[0], args.system_prompt)
        full_prompts_1 = apply_chat_template(tokenizer, df_anchors_list[1], args.system_prompt)
        direction_vectors = {}
        for layer in args.intervention_layers:
            h0 = get_hidden_states(model, tokenizer, full_prompts_0, layer, layer)[0]
            h1 = get_hidden_states(model, tokenizer, full_prompts_1, layer, layer)[0]
            diff = compute_difference_vectors(h0, h1).mean(dim=0)
            direction_vectors[layer] = diff.to(model.device)
        # 如需要保存
        if args.save_directions:
            with open(args.save_directions, "w") as f:
                json.dump({k: v.tolist() for k, v in direction_vectors.items()}, f, indent=2)
            print(f"[Save] The difference vector has been saved to{args.save_directions}")

    # 加载 prompts 数据
    df_texts_list = []
    for idx, path in enumerate(args.datasets):
        col = args.text_columns[idx] if args.text_columns is not None else None
        df = load_dataset(path, column_name=col)
        df = df.sample(min(args.num_samples, len(df)), random_state=args.seed)
        df_texts_list.append(df)

    df = df_texts_list[0]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    prompts = df[args.text_columns[0]].tolist() if args.text_columns is not None else df.iloc[:, 0].tolist()

    evaluate_and_save_outputs(model, tokenizer, prompts, direction_vectors, args.intervention_layers, args.separation_file, strength_dict)


if __name__ == "__main__":
    main()
