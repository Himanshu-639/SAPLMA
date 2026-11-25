"""
Optimized Activation Extraction for All 28 Layers of LLaMA-3.2-3B.
Extracts hidden state activations for all 28 transformer layers simultaneously
in a single forward pass per statement across the 6 True-False topics.

Supports:
- Precision: fp16, 8bit (LLM.int8), 4bit (NF4)
- Models: LLaMA-3.2-3B (Base) and LLaMA-3.2-3B-Instruct
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# LLaMA-3.2-3B has 28 hidden layers
NUM_LAYERS = 28
ALL_LAYERS = [-i for i in range(1, NUM_LAYERS + 1)]  # [-1, -2, ..., -28]
TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

def get_hf_token():
    token = os.getenv("HF_TOKEN", None)
    if not token:
        env_path = os.path.join(SCRIPT_DIR, "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "HF_TOKEN" in line and "=" in line:
                        token = line.split("=")[1].strip().strip('"').strip("'")
                        break
    return token

def load_model_and_tokenizer(model_type="base", precision="fp16"):
    token = get_hf_token()
    if model_type == "instruct":
        hf_id = "meta-llama/Llama-3.2-3B-Instruct"
        base_tag = "llama3.2_3b_instruct"
    else:
        hf_id = "meta-llama/Llama-3.2-3B"
        base_tag = "llama3.2_3b"

    if precision == "8bit":
        model_tag = f"{base_tag}_8bit"
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading {hf_id} in 8-bit (LLM.int8)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=quant_config,
            device_map="auto",
            token=token
        )
    elif precision == "4bit":
        model_tag = f"{base_tag}_4bit"
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(f"Loading {hf_id} in 4-bit (NF4, compute=bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=quant_config,
            device_map="auto",
            token=token
        )
    else:
        model_tag = base_tag
        print(f"Loading {hf_id} in Full Precision (FP16)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=token
            )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, model_tag

def extract_activations_all_layers(
    model_type="base",
    precision="fp16",
    data_dir="data",
    output_dir="embeddings"
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, model_tag = load_model_and_tokenizer(model_type, precision)
    device = next(model.parameters()).device

    print("\n" + "=" * 80)
    print(f" EXTRACTING ACTIVATIONS FOR ALL 28 LAYERS: {model_tag} ({precision})")
    print(f" Device: {device} | Output Directory: {output_dir}")
    print("=" * 80 + "\n")

    with torch.no_grad():
        for dataset_name in TOPICS:
            csv_path = os.path.join(data_dir, f"{dataset_name}_true_false.csv")
            if not os.path.exists(csv_path):
                if dataset_name == "cities" and os.path.exists(os.path.join(data_dir, "capitals_true_false.csv")):
                    csv_path = os.path.join(data_dir, "capitals_true_false.csv")
                else:
                    print(f"Warning: File {csv_path} not found. Skipping.")
                    continue

            df = pd.read_csv(csv_path)
            num_sentences = len(df)
            print(f"Processing '{dataset_name}' ({num_sentences} sentences across 28 layers)...")

            # Pre-allocate dictionary of lists for each layer
            layer_embeddings = {abs(layer): [] for layer in ALL_LAYERS}
            next_ids = []

            t_topic_start = time.time()
            for i, row in df.iterrows():
                prompt = str(row['statement']).rstrip(". ")
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                outputs = model(**inputs, output_hidden_states=True)
                # outputs.hidden_states has 29 tensors: index 0 is embedding, 1..28 are transformer layers
                # In PyTorch: outputs.hidden_states[layer] with negative index -1 is last layer, -28 is first layer
                for layer in ALL_LAYERS:
                    last_token_vec = outputs.hidden_states[layer][0, -1].float().cpu().numpy().tolist()
                    layer_embeddings[abs(layer)].append([last_token_vec])

                # Get next token prediction id
                next_id = outputs.logits[0, -1].argmax().item()
                next_ids.append(float(next_id))

                if (i + 1) % 250 == 0 or (i + 1) == num_sentences:
                    elapsed = time.time() - t_topic_start
                    rate = (i + 1) / elapsed
                    print(f"  Processed {i+1}/{num_sentences} sentences ({rate:.1f} sent/s)")

            # Save CSV for each of the 28 layers
            for layer in ALL_LAYERS:
                layer_num = abs(layer)
                out_df = df.copy()
                out_df['embeddings'] = layer_embeddings[layer_num]
                out_df['next_id'] = next_ids

                out_path = os.path.join(output_dir, f"embeddings_with_labels_{dataset_name}{model_tag}_{layer_num}_rmv_period.csv")
                out_df.to_csv(out_path, index=False)

            print(f"  --> Saved 28 layer files for '{dataset_name}'.")

    total_time = time.time() - start_time
    print(f"\n[Extraction Completed] Total time: {total_time/60:.2f} minutes for {model_tag}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations for all 28 layers")
    parser.add_argument("--model", type=str, default="base", choices=["base", "instruct"])
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "8bit", "4bit"])
    args = parser.parse_args()

    extract_activations_all_layers(
        model_type=args.model,
        precision=args.precision,
        data_dir=os.path.join(SCRIPT_DIR, "data"),
        output_dir=os.path.join(SCRIPT_DIR, "embeddings")
    )
