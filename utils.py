import ast
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
  AutoModelForCausalLM,
AutoTokenizer,
)


def set_seed(seed):
    """
        Set random seeds for reproducibility of experiments.

        Args:
            seed: The seed value to use for all random number generators
        """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(model_name):
    """
       Load pre-trained causal language model and corresponding tokenizer.

       Args:
           model_path: Path or identifier of the pre-trained model
           load_in_4bit: Whether to load model in 4-bit quantization
           device: Device to load the model onto (e.g., "cuda" or "cpu")

       Returns:
           Tuple containing the loaded model and tokenizer
       """
    is_higher_than_ampere = torch.cuda.is_bf16_supported()
    try:
        import flash_attn

        is_flash_attn_available = True
    except:
        is_flash_attn_available = False


    if is_higher_than_ampere and is_flash_attn_available:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        print("Using FP16 and Flash-Attention 2...")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Using FP16 and normal attention implementation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.clean_up_tokenization_spaces == True:
        print(
            "WARNING: tokenizer.clean_up_tokenization_spaces is by default set to True. "
            "This will break the attack when validating re-tokenization invariance. Setting it to False..."
        )
        tokenizer.clean_up_tokenization_spaces = False

     if not tokenizer.chat_template:
        print(
            f"The tokenizer of {model_name} does not come with a chat template. Dynamically setting one..."
        )
        if "HarmBench-Llama-2-13b-cls" in model_name:
            # If you are sure that the model does not require a chat template, you can skip this step like this
            print(
                "HarmBench-Llama-2-13b-cls does not require a chat template. Skipped."
            )
        else:
            raise ValueError(
                f"The chat template for the tokenizer of {model_name} is not available. "
                "To avoid unexpected behavior, it cannot proceed with the default chat template. "
                "Please implement it manually in `load_model_and_tokenizer()`, `utils.py`."
            )
    return model, tokenizer


def load_dataset(dataset_path, column_name=None) -> pd.Series:
    """
       Load dataset from CSV or TXT file.

       Args:
           file_path: Path to the dataset file
           columns: Specific columns to load from CSV (None for first column)

       Returns:
           pandas Series containing the loaded data
       """
    _, file_extension = os.path.splitext(dataset_path)

    if file_extension.lower() == ".csv":
        if column_name is not None:
            df = pd.read_csv(dataset_path)
            df = df.dropna(subset=[column_name])
            return df[column_name]
        else:
            df = pd.read_csv(dataset_path, header=None)
            df = df.dropna(subset=[0])
            return df[0]
    elif file_extension.lower() == ".txt":
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = file.read().splitlines()
        df = pd.DataFrame(data, columns=["source"])
        return df["source"]
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")






