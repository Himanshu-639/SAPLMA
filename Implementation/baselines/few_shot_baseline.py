"""
Few-Shot Prompting Baseline (3-shot and 5-shot)
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 5:
"Namely, we provided few statements along with the ground-truth label, and then
the statement in question. We recorded the probability that the LLM assigned to the
token 'true' and to the token 'false'. Unfortunately, the LLM had a tendency to
assign higher values to the 'true' token; therefore, we divided the probability
assigned to the 'true' token by the one assigned to the 'false' token. Finally,
we considered the LLM's prediction 'true' if the value was greater than the
average, and 'false' otherwise. We tested a 3-shot and a 5-shot version."
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

def format_example(statement: str, label: int) -> str:
    stmt = statement.rstrip(". ")
    label_str = "true" if int(label) == 1 else "false"
    return f"{stmt}: {label_str}. "

def evaluate_few_shot_topic(
    df: pd.DataFrame,
    model,
    tokenizer,
    k_shots: int = 3,
    device: str = "cuda"
) -> float:
    true_token_id = tokenizer.encode(" true")[-1] if " true" in tokenizer.get_vocab() else tokenizer.encode("true")[-1]
    false_token_id = tokenizer.encode(" false")[-1] if " false" in tokenizer.get_vocab() else tokenizer.encode("false")[-1]

    # Pre-select k representative examples from the beginning for demonstration
    demo_indices = list(range(k_shots))
    demo_prompt = "".join([format_example(df.at[j, 'statement'], df.at[j, 'label']) for j in demo_indices])

    ratios = []
    labels = []

    # Evaluate on the remaining statements
    for i in range(k_shots, len(df)):
        test_stmt = str(df.at[i, 'statement']).rstrip(". ")
        full_prompt = demo_prompt + f"{test_stmt}: "

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs.input_ids)
            last_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            p_true = probs[true_token_id].item()
            p_false = probs[false_token_id].item()

            ratio = p_true / (p_false + 1e-12)
            ratios.append(ratio)
            labels.append(int(df.at[i, 'label']))

    ratios = np.array(ratios)
    labels = np.array(labels)
    # Normalized by topic average ratio as described in the paper
    avg_ratio = np.mean(ratios)
    predictions = (ratios > avg_ratio).astype(int)
    acc = np.mean(predictions == labels)
    return acc

def run_few_shot_baseline(
    model_name: str = "facebook/opt-6.7b",
    k_shots_list: list = [3, 5],
    data_dir: str = "data",
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Few-Shot Prompting Baseline on {model_name} (Device: {device})")
    print(f"=======================================================\n")

    try:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []
    for k in k_shots_list:
        print(f"\nEvaluating {k}-shot baseline across all 6 topics:")
        topic_accs = {}
        for topic in TOPICS:
            p = os.path.join(data_dir, f"{topic}_true_false.csv")
            df = pd.read_csv(p)
            acc = evaluate_few_shot_topic(df, model, tokenizer, k_shots=k, device=str(device))
            topic_accs[topic] = acc
            print(f"  [{topic:<12}] Acc ({k}-shot) = {acc:.4f}")

        avg_acc = np.mean(list(topic_accs.values()))
        print(f"  --> Average {k}-shot Accuracy: {avg_acc:.4f}")
        row = {"Shots": k, **topic_accs, "Average": avg_acc}
        results.append(row)

    out_df = pd.DataFrame(results)
    out_file = os.path.join(output_dir, "few_shot_baseline_results.csv")
    out_df.to_csv(out_file, index=False)
    print(f"\nSaved results to: {out_file}\n")
    print(out_df.to_string(index=False))
    return out_df

if __name__ == "__main__":
    run_few_shot_baseline()
