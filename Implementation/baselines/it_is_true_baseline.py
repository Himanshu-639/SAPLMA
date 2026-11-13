"""
"It-is-true that X" vs. "It-is-false that X" Baseline
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 5:
"The third baseline, given a statement 'X', we measure the probabilities (using OPT-6.7)
of the sentences 'It is true that X', and 'It is false that X', and pick the higher
probability (considering only the probabilities for X, not the added words).
This normalizes the length and frequency factors."
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

def compute_conditional_statement_prob(prefix: str, statement: str, model, tokenizer, device="cuda") -> float:
    """
    Computes the product of conditional probabilities for tokens in `statement`,
    given `prefix`. (Only probabilities of statement X are multiplied).
    """
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    full_text = prefix + statement
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

    input_ids = torch.tensor([full_tokens]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, :-1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Slice corresponding to the tokens of statement X only
        start_idx = len(prefix_tokens) - 1
        target_tokens = full_tokens[len(prefix_tokens):]

        if len(target_tokens) == 0:
            return 1.0

        target_ids = torch.tensor(target_tokens).to(device)
        positions = torch.arange(start_idx, start_idx + len(target_tokens)).to(device)
        selected_log_probs = log_probs[positions, target_ids]
        statement_log_prob = selected_log_probs.sum().item()

    return statement_log_prob

def run_it_is_true_baseline(
    model_name: str = "facebook/opt-6.7b",
    data_dir: str = "data",
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" 'It-is-true' Baseline Evaluation on {model_name} (Device: {device})")
    print(f"=======================================================\n")

    try:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    topic_accuracies = {}
    for topic in TOPICS:
        p = os.path.join(data_dir, f"{topic}_true_false.csv")
        df = pd.read_csv(p)
        print(f"Evaluating topic '{topic}' ({len(df)} statements)...")

        correct = 0
        total = 0
        for i, row in df.iterrows():
            stmt = str(row['statement'])
            label = int(row['label'])

            lp_true = compute_conditional_statement_prob("It is true that ", stmt, model, tokenizer, device=str(device))
            lp_false = compute_conditional_statement_prob("It is false that ", stmt, model, tokenizer, device=str(device))

            pred = 1 if lp_true > lp_false else 0
            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(df):
                print(f"  Processed {i+1}/{len(df)}: Running Acc = {correct/total:.4f}")

        acc = correct / total
        topic_accuracies[topic] = acc
        print(f"  [{topic:<12}] Acc = {acc:.4f}")

    avg_acc = np.mean(list(topic_accuracies.values()))
    print(f"\nAverage 'It-is-true' Baseline Accuracy: {avg_acc:.4f}")
    print(f"Paper Table 1 Reported Average: 0.5593")

    out_df = pd.DataFrame([{
        "Topic": t,
        "Accuracy": topic_accuracies[t]
    } for t in TOPICS] + [{"Topic": "Average", "Accuracy": avg_acc}])
    out_file = os.path.join(output_dir, "it_is_true_baseline_results.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved results to: {out_file}\n")
    return out_df

if __name__ == "__main__":
    run_it_is_true_baseline()
