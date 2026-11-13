"""
Activation Extraction Module
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 4 & 7:
- Records hidden layer activations from candidate layers [-1, -4, -8, -12, -16]
- Extracts the hidden state vector corresponding to the final token of each statement.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, BitsAndBytesConfig
from typing import List, Dict

# Standard candidate layers evaluated in the paper
DEFAULT_LAYERS = [-1, -4, -8, -12, -16]

# Standard 6 True-False topics from Section 3
TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

def get_model_and_tokenizer(model_name: str, precision: str = "fp16", device: str = "cuda"):
    """Loads LLM model and tokenizer in fp16, 8-bit, or 4-bit precision."""
    token = os.getenv("HF_TOKEN", None)
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "HF_TOKEN" in line and "=" in line:
                        token = line.split("=")[1].strip().strip('"').strip("'")
                        break

    quantization_config = None
    if precision == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization: BitsAndBytesConfig(load_in_8bit=True)")
    elif precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("Using 4-bit quantization: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)")
    
    if "opt" in model_name.lower():
        hf_id = "facebook/opt-6.7b" if "6.7" in model_name else model_name
        if quantization_config is not None:
            model = OPTForCausalLM.from_pretrained(
                hf_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=token
            )
        else:
            try:
                model = OPTForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, local_files_only=True).to(device)
            except Exception:
                model = OPTForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, token=token).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    else:
        if "instruct" in model_name.lower():
            hf_id = "meta-llama/Llama-3.2-3B-Instruct"
        elif "llama-2" in model_name.lower():
            hf_id = "meta-llama/Llama-2-7b-hf"
        elif "3" in model_name.lower() or "llama" in model_name.lower():
            hf_id = "meta-llama/Llama-3.2-3B"
        else:
            hf_id = model_name

        if quantization_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=token
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, local_files_only=True).to(device)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, token=token).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)

    return model, tokenizer

def extract_and_save_embeddings(
    model_name: str,
    precision: str = "fp16",
    datasets: List[str] = None,
    data_dir: str = "data",
    output_dir: str = "embeddings",
    layers_to_use: List[int] = None,
    remove_period: bool = True
):
    if datasets is None:
        datasets = TOPICS
    if layers_to_use is None:
        layers_to_use = DEFAULT_LAYERS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting activations on: {device} (Precision: {precision})")
    model, tokenizer = get_model_and_tokenizer(model_name, precision=precision, device=str(device))

    if "instruct" in model_name.lower():
        base_tag = "llama3.2_3b_instruct"
    elif "opt" in model_name.lower():
        base_tag = "6.7b"
    elif "3" in model_name:
        base_tag = "llama3.2_3b"
    else:
        base_tag = "LLAMA7"

    if precision in ["8bit", "4bit"]:
        model_tag = f"{base_tag}_{precision}"
    else:
        model_tag = base_tag

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for dataset_name in datasets:
            csv_path = os.path.join(data_dir, f"{dataset_name}_true_false.csv")
            if not os.path.exists(csv_path):
                # Try fallback for cities/capitals
                if dataset_name == "cities" and os.path.exists(os.path.join(data_dir, "capitals_true_false.csv")):
                    csv_path = os.path.join(data_dir, "capitals_true_false.csv")
                else:
                    print(f"Warning: File {csv_path} not found. Skipping.")
                    continue

            df = pd.read_csv(csv_path)
            print(f"Processing '{dataset_name}' ({len(df)} sentences)...")

            dfs: Dict[int, pd.DataFrame] = {layer: df.copy() for layer in layers_to_use}
            for layer in layers_to_use:
                dfs[layer]['embeddings'] = pd.Series(dtype='object')
                dfs[layer]['next_id'] = pd.Series(dtype=float)

            for i, row in df.iterrows():
                prompt = str(row['statement'])
                if remove_period:
                    prompt = prompt.rstrip(". ")
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                outputs = model.generate(
                    inputs.input_ids,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                    min_new_tokens=1
                )
                next_id = np.array(outputs[0].cpu())[0][-1]
                for layer in layers_to_use:
                    last_hidden_state = outputs.hidden_states[0][layer][0][-1].float().cpu()
                    dfs[layer].at[i, 'embeddings'] = [last_hidden_state.numpy().tolist()]
                    dfs[layer].at[i, 'next_id'] = next_id

                if i % 200 == 0 or i == len(df) - 1:
                    print(f"  Processed {i+1}/{len(df)} sentences")

            for layer in layers_to_use:
                layer_suffix = "_rmv_period.csv" if remove_period else ".csv"
                out_path = os.path.join(output_dir, f"embeddings_with_labels_{dataset_name}{model_tag}_{abs(layer)}{layer_suffix}")
                dfs[layer].to_csv(out_path, index=False)
                print(f"  Saved: {out_path}")

    print("Activation extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LLM hidden state activations for SAPLMA.")
    parser.add_argument("--model", type=str, default="opt-6.7b", choices=["opt-6.7b", "llama3.2-3b", "llama-2-7b"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    args = parser.parse_args()

    extract_and_save_embeddings(args.model, data_dir=args.data_dir, output_dir=args.output_dir)
