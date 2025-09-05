import json
import argparse
import os
import random
from itertools import cycle
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
    parser = argparse.ArgumentParser(description="Test intervention effects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")#Select the path to the model you are testing.
    parser.add_argument("--anchors", type=str, nargs="+", default=["./data/benigntest.txt", "./data/harmfultest.txt"])#anchor datasets path
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--text_columns", type=str, nargs="+")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--system_prompt", type=str)
    parser.add_argument("--separation_file", type=str, required=True)
    parser.add_argument("--intervention_layers", type=int, nargs="+", required=True)
    parser.add_argument("--intervention_strength", type=float, default=1)
    return parser.parse_args()


def apply_chat_template(tokenizer, texts, system_prompt=None):
    full_prompt_list = []
    for text in texts:
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        else:
            messages = [{"role": "user", "content": text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt_list.append(full_prompt)
    return full_prompt_list


def get_hidden_states(model, tokenizer, full_prompt_list, target_layers):
    model.eval()
    all_layer_hidden_states = {layer: [] for layer in target_layers}

    with torch.no_grad():
        for full_prompt in tqdm(full_prompt_list, desc="Calculating hidden states"):
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            for layer in target_layers:
                all_layer_hidden_states[layer].append(outputs.hidden_states[layer][:, -1, :])

    for layer in target_layers:
        all_layer_hidden_states[layer] = torch.stack(all_layer_hidden_states[layer])
    return all_layer_hidden_states


def compute_difference_vectors(harmless_embeddings, harmful_embeddings):
    difference_vectors = []
    for harmless, harmful in zip(harmless_embeddings, harmful_embeddings):
        diff = harmful - harmless
        difference_vectors.append(diff)
    return torch.stack(difference_vectors)


def register_hooks(model, intervention_layers, direction_vectors, intervention_strength):
    hooks = []

    def hook(module, input, output, layer_idx):
        direction = direction_vectors.get(layer_idx)
        if direction is not None:
            output[0][:, -1, :] -= intervention_strength * direction.squeeze(0)
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
                                        intervention_strength, max_new_tokens=256):
    model.eval()
    with torch.no_grad():
        full_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        hooks = register_hooks(model, intervention_layers, direction_vectors, intervention_strength)

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


def evaluate_and_save_outputs(model, tokenizer, prompts, direction_vectors, intervention_layers, save_path, intervention_strength):
    results = []

    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        no_intv_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

        inputs = tokenizer(no_intv_prompt, return_tensors="pt", padding=True, return_length=True).to(model.device)
        input_length = inputs['length'][0].item()

        no_intv_output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=256,
            do_sample=False
        )
        no_intv_text = tokenizer.decode(no_intv_output[0][input_length:], skip_special_tokens=True)

        intv_text = generate_response_with_intervention(model, tokenizer, prompt, direction_vectors, intervention_layers, intervention_strength)

        results.append({
            "prompt": prompt,
            "response(no intervention)": no_intv_text.strip(),
            "response(intervention)": intv_text.strip()
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"The results have been saved to {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    intervention_strength = args.intervention_strength
    intervention_layers = args.intervention_layers

    df_anchors_list = [load_dataset(path) for path in args.anchors]

    df_texts_list = []
    for idx, dataset_path in enumerate(args.datasets):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        column_name = args.text_columns[idx] if args.text_columns is not None else None
        df_texts = load_dataset(dataset_path, column_name=column_name)
        df_texts = df_texts.sample(min(args.num_samples, len(df_texts)), random_state=args.seed)
        df_texts_list.append(df_texts)

    model, tokenizer = load_model_and_tokenizer(args.model)

    direction_vectors = {}
    for layer in intervention_layers:
        full_prompts_0 = apply_chat_template(tokenizer, df_anchors_list[0], args.system_prompt)
        full_prompts_1 = apply_chat_template(tokenizer, df_anchors_list[1], args.system_prompt)
        h0 = get_hidden_states(model, tokenizer, full_prompts_0, [layer])[layer]
        h1 = get_hidden_states(model, tokenizer, full_prompts_1, [layer])[layer]
        diff = compute_difference_vectors(h0, h1).mean(dim=0)
        direction_vectors[layer] = diff

    df = df_texts_list[0]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    prompts = df[args.text_columns[0]].tolist() if args.text_columns is not None else df.iloc[:, 0].tolist()

    evaluate_and_save_outputs(
        model, tokenizer,
        prompts=prompts,
        direction_vectors=direction_vectors,
        intervention_layers=intervention_layers,
        save_path=args.separation_file,
        intervention_strength=intervention_strength
    )


if __name__ == "__main__":
    main()
