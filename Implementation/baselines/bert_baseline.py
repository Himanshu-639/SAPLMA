"""
BERT Sentence Embedding Baseline
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 5:
"The first baseline is BERT, for which we train a classifier (with an identical
architecture to the one used by SAPLMA) on the BERT embeddings of each sentence."
Protocol:
- Mean-pooling over BERT last hidden states.
- 6-fold leave-one-topic-out cross-validation across all 6 topics.
- Identical 3-layer MLP: Dense(256) -> Dense(128) -> Dense(64) -> Dense(1, sigmoid).
- Fixed threshold of 0.5.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_curve, auc, accuracy_score
from saplma.model import build_saplma_classifier

TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

def get_bert_embeddings(statements, tokenizer, model, device="cuda"):
    """Computes mean-pooled BERT representations."""
    embeddings = []
    batch_size = 64
    for i in range(0, len(statements), batch_size):
        batch = [str(s).rstrip(". ") for s in statements[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean-pooling across tokens
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

def run_bert_baseline(
    data_dir: str = "data",
    output_dir: str = "results",
    repeat_each: int = 3,
    epochs: int = 5,
    batch_size: int = 32
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" BERT Baseline Evaluation (Device: {device})")
    print(f" Protocol: Leave-One-Topic-Out, Repetitions={repeat_each}, Epochs={epochs}, Threshold=0.5")
    print(f"=======================================================\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()

    topic_data = {}
    for topic in TOPICS:
        p = os.path.join(data_dir, f"{topic}_true_false.csv")
        df = pd.read_csv(p)
        print(f"Extracting BERT embeddings for '{topic}' ({len(df)} sentences)...")
        embs = get_bert_embeddings(df['statement'].tolist(), tokenizer, bert_model, device=str(device))
        topic_data[topic] = {
            'X': embs,
            'y': np.array(df['label'], dtype=int)
        }

    summary = {}
    for test_idx, test_topic in enumerate(TOPICS):
        test_X = topic_data[test_topic]['X']
        test_y = topic_data[test_topic]['y']

        train_X = np.vstack([topic_data[t]['X'] for t in TOPICS if t != test_topic])
        train_y = np.concatenate([topic_data[t]['y'] for t in TOPICS if t != test_topic])

        run_accs = []
        run_aucs = []
        for r in range(repeat_each):
            classifier = build_saplma_classifier(input_dim=train_X.shape[1])
            classifier.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

            preds = classifier.predict(test_X, verbose=0).flatten()
            acc = accuracy_score(test_y, (preds > 0.5).astype(int))
            run_accs.append(acc)

            fpr, tpr, _ = roc_curve(test_y, preds)
            run_aucs.append(auc(fpr, tpr))

        mean_acc = np.mean(run_accs)
        mean_auc = np.mean(run_aucs)
        summary[test_topic] = (mean_acc, mean_auc)
        print(f"  [{test_topic:<12}] Acc = {mean_acc:.4f}, AUC = {mean_auc:.4f}")

    avg_acc = np.mean([summary[t][0] for t in TOPICS])
    avg_auc = np.mean([summary[t][1] for t in TOPICS])
    print(f"\nBERT Baseline Average Accuracy: {avg_acc:.4f} (Avg AUC: {avg_auc:.4f})")
    print(f"Paper Table 1 BERT Average: 0.5434")

    out_df = pd.DataFrame([{
        "Topic": t,
        "Accuracy": summary[t][0],
        "AUC": summary[t][1]
    } for t in TOPICS] + [{"Topic": "Average", "Accuracy": avg_acc, "AUC": avg_auc}])
    out_file = os.path.join(output_dir, "bert_baseline_results.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Results saved to: {out_file}\n")
    return out_df

if __name__ == "__main__":
    run_bert_baseline()
