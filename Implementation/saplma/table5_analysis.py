"""
Statement Probability vs. SAPLMA Analysis (Replicating Table 5)
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 6:
Compares the probability assigned by the LLM and the sigmoid output from SAPLMA's
28th layer (using OPT-6.7b) on statements designed to expose token frequency
and sentence-length vulnerabilities in pure language model probabilities.
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, OPTForCausalLM
from saplma.model import build_saplma_classifier, parse_embedding_str
from saplma.evaluate_cross_topic import load_topic_dataset, TOPICS

TABLE_5_STATEMENTS = [
    ("H2O is water, which is essential for humans", True),
    ("Humans don't need water", False),
    ("The sun is hot, and it radiates its heat to Earth", True),
    ("The sun protects Earth from heat", False),
    ("The Earth is flat", False),
    ("The world is round and rotates", True),
    ("The Earth is flat like a pancake", False),
    ("Kevin Durant is a basketball player", True),
    ("Kevin Durant is a baseball player", False),
    ("Kevin Durant is basketeer", True),
    ("Kevin Duarnt is basketball player", True),
    ("Jennifer Aniston is an actress", True),
    ("Jennifer Aniston is not an actress", False),
    ("Jennifer Aniston is a female person", True),
    ("Harry Potter is real", False),
    ("Harry Potter is fictional", True),
    ("Harry Potter is an imaginary figure", True),
]

def compute_sentence_prob(sentence: str, model, tokenizer, device="cuda") -> float:
    """Computes total sentence probability: product of token conditional probabilities."""
    tokens = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[:, :-1, :]
        targets = tokens[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        sentence_prob = torch.exp(token_log_probs.sum()).item()
    return sentence_prob

def run_table5_analysis(
    embeddings_dir: str = "embeddings",
    model_name: str = "facebook/opt-6.7b",
    layer: int = -4, # 28th layer
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Replicating Table 5: Sentence Probability vs SAPLMA 28th-layer")
    print(f"=======================================================\n")

    # 1. Train SAPLMA 28th-layer classifier on the entire True-False dataset
    print("Training SAPLMA 28th-layer classifier on True-False dataset...")
    train_dfs = [load_topic_dataset(embeddings_dir, t, "6.7b", layer) for t in TOPICS]
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_X = np.array([parse_embedding_str(e) for e in train_df['embeddings']])
    train_y = np.array(train_df['label'], dtype=int)

    classifier = build_saplma_classifier(input_dim=train_X.shape[1])
    classifier.fit(train_X, train_y, epochs=5, batch_size=32, verbose=0)
    print("Classifier trained successfully.")

    # 2. Load model for sentence probability and activation extraction
    try:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []
    print("\nEvaluating statements:")
    for stmt, label in TABLE_5_STATEMENTS:
        # A. Sentence probability
        prob = compute_sentence_prob(stmt, model, tokenizer, device=str(device))

        # B. Hidden state at 28th layer
        prompt = stmt.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
                min_new_tokens=1
            )
            act = outputs.hidden_states[0][layer][0][-1].cpu().numpy()

        # C. Classifier sigmoid prediction
        saplma_score = classifier.predict(np.expand_dims(act, axis=0), verbose=0)[0][0]

        results.append({
            "Statement": stmt,
            "Label": "True" if label else "False",
            "LLM_Probability": f"{prob:.2e}",
            "SAPLMA_Score": round(float(saplma_score), 4)
        })

    df_out = pd.DataFrame(results)
    out_file = os.path.join(output_dir, "table5_statement_comparison.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved Table 5 results to: {out_file}\n")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    run_table5_analysis()
