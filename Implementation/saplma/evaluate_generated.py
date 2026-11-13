"""
Evaluation on LLM-Generated Statements (Replicating Table 3 and Table 4)
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 5:
"Each of the models was trained 14 times using the same classifier described in Section 4.
The models were trained on the entire true-false dataset (i.e., all topics, but not
the generated sentences) and tested on the generated sentences."
- Table 3: Threshold = 0.5.
- Table 4: Threshold estimated from a 30% held-out validation set, tested on remaining 70%.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from saplma.model import build_saplma_classifier, parse_embedding_str
from saplma.evaluate_cross_topic import load_topic_dataset, TOPICS, LAYERS, LAYER_NAMES

def evaluate_on_generated(
    model_name: str = "opt",
    embeddings_dir: str = "embeddings",
    output_dir: str = "results",
    repeat_each: int = 14,
    epochs: int = 5,
    batch_size: int = 32
):
    model_tag = "6.7b" if "opt" in model_name.lower() else ("llama3.2_3b" if "3" in model_name else "LLAMA7")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" SAPLMA LLM-Generated Statements Evaluation: {model_name} (tag={model_tag})")
    print(f" Protocol: Train on all 6 topics, Test on 245 generated sentences, Repetitions={repeat_each}")
    print(f"=======================================================\n")

    summary_rows_tab3 = []
    summary_rows_tab4 = []

    for layer in LAYERS:
        layer_label = LAYER_NAMES.get(layer, f"layer_{layer}")
        print(f"\n>>> Evaluating {layer_label} (index {layer}) <<<")

        # 1. Load all 6 topics as training set
        train_dfs = [load_topic_dataset(embeddings_dir, t, model_tag, layer) for t in TOPICS]
        train_df = pd.concat(train_dfs, ignore_index=True)

        # 2. Load the 245 generated sentences
        gen_file = os.path.join(embeddings_dir, f"embeddings_with_labels_generated{model_tag}_{abs(layer)}_rmv_period.csv")
        if not os.path.exists(gen_file):
            gen_file = os.path.join(embeddings_dir, f"embeddings_with_labels_generated{model_tag}_{abs(layer)}.csv")
        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Generated statements embeddings not found at {gen_file}. Run extract_activations.py first.")
        test_df = pd.read_csv(gen_file)

        train_X = np.array([parse_embedding_str(e) for e in train_df['embeddings']])
        train_y = np.array(train_df['label'], dtype=int)
        test_X = np.array([parse_embedding_str(e) for e in test_df['embeddings']])
        test_y = np.array(test_df['label'], dtype=int)

        t3_accs, t3_aucs = [], []
        t4_thresholds, t4_accs = [], []

        for r in range(repeat_each):
            model = build_saplma_classifier(input_dim=train_X.shape[1])
            model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

            test_preds = model.predict(test_X, verbose=0).flatten()

            # --- Table 3: Threshold = 0.5 ---
            b_preds = (test_preds > 0.5).astype(int)
            t3_accs.append(accuracy_score(test_y, b_preds))
            fpr, tpr, _ = roc_curve(test_y, test_preds)
            t3_aucs.append(auc(fpr, tpr))

            # --- Table 4: 30% Validation Set for Threshold Estimation, 70% Test ---
            val_X, eval_X, val_y, eval_y = train_test_split(
                test_X, test_y, test_size=0.7, random_state=42 + r
            )
            val_preds = model.predict(val_X, verbose=0).flatten()
            fpr_v, tpr_v, thresh_v = roc_curve(val_y, val_preds)
            best_thresh = thresh_v[np.argmax([accuracy_score(val_y, val_preds > thr) for thr in thresh_v])]

            eval_preds = model.predict(eval_X, verbose=0).flatten()
            eval_binary = (eval_preds > best_thresh).astype(int)
            t4_thresholds.append(best_thresh)
            t4_accs.append(accuracy_score(eval_y, eval_binary))

        mean_t3_acc = np.mean(t3_accs)
        mean_t3_auc = np.mean(t3_aucs)
        mean_t4_thresh = np.mean(t4_thresholds)
        mean_t4_acc = np.mean(t4_accs)

        print(f"  Table 3 (Thresh 0.5): Acc = {mean_t3_acc:.4f}, AUC = {mean_t3_auc:.4f}")
        print(f"  Table 4 (Optimized):  Avg Thresh = {mean_t4_thresh:.4f}, Acc = {mean_t4_acc:.4f}")

        summary_rows_tab3.append({
            "Layer": layer_label,
            "Accuracy": mean_t3_acc,
            "AUC": mean_t3_auc
        })
        summary_rows_tab4.append({
            "Layer": layer_label,
            "Avg_Threshold": mean_t4_thresh,
            "Accuracy": mean_t4_acc
        })

    df3 = pd.DataFrame(summary_rows_tab3)
    df4 = pd.DataFrame(summary_rows_tab4)

    file_tab3 = os.path.join(output_dir, f"table3_generated_fixed_threshold_{model_tag}.csv")
    file_tab4 = os.path.join(output_dir, f"table4_generated_optimal_threshold_{model_tag}.csv")
    df3.to_csv(file_tab3, index=False)
    df4.to_csv(file_tab4, index=False)

    print(f"\nSaved Table 3 results to: {file_tab3}")
    print(df3.to_string(index=False))
    print(f"\nSaved Table 4 results to: {file_tab4}")
    print(df4.to_string(index=False))

    return df3, df4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAPLMA on LLM-generated statements.")
    parser.add_argument("--model", type=str, default="opt", choices=["opt", "llama3", "llama2"])
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--rep", type=int, default=14, help="Number of training runs (paper specifies 14).")
    args = parser.parse_args()

    evaluate_on_generated(
        model_name=args.model,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        repeat_each=args.rep
    )
