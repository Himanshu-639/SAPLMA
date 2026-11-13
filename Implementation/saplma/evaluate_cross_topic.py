"""
Cross-Topic Leave-One-Out Evaluation (Replicating Table 1 and Table 2)
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 4 & 5:
"For each topic in the true-false dataset, we train the classifier using only the
activation values obtained from all other topics and test its accuracy on the
current topic... To obtain more reliable results, we train each classifier three
times with different initial random weights. This process is repeated for each topic,
and we report the accuracy mean over these three runs."
Standard decision threshold is 0.5.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from saplma.model import build_saplma_classifier, parse_embedding_str

TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]
LAYERS = [-1, -4, -8, -12, -16]

LAYER_NAMES = {
    -1: "last-layer",
    -4: "28th-layer",
    -8: "24th-layer",
    -12: "20th-layer",
    -16: "middle-layer"
}

def load_topic_dataset(embeddings_dir: str, topic: str, model_tag: str, layer: int) -> pd.DataFrame:
    """Loads pre-extracted activations, falling back between cities and capitals if needed."""
    candidate_names = [topic]
    if topic == "cities":
        candidate_names.append("capitals")
    elif topic == "capitals":
        candidate_names.append("cities")

    for name in candidate_names:
        p = os.path.join(embeddings_dir, f"embeddings_with_labels_{name}{model_tag}_{abs(layer)}_rmv_period.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
        p_no_rmv = os.path.join(embeddings_dir, f"embeddings_with_labels_{name}{model_tag}_{abs(layer)}.csv")
        if os.path.exists(p_no_rmv):
            return pd.read_csv(p_no_rmv)

    raise FileNotFoundError(f"Could not find embeddings file for topic '{topic}' (model={model_tag}, layer={layer}) in '{embeddings_dir}'.")

def run_cross_topic_evaluation(
    model_name: str = "opt",
    precision: str = "fp16",
    embeddings_dir: str = "embeddings",
    output_dir: str = "results",
    repeat_each: int = 3,
    epochs: int = 5,
    batch_size: int = 32
):
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
    elif "_8bit" in model_name.lower() or "_4bit" in model_name.lower():
        model_tag = model_name
    else:
        model_tag = base_tag

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" SAPLMA Cross-Topic Evaluation: {model_name} (precision={precision}, tag={model_tag})")
    print(f" Protocol: 6-fold Leave-One-Topic-Out, Repetitions={repeat_each}, Epochs={epochs}, Threshold=0.5")
    print(f"=======================================================\n")

    summary_rows = []

    for layer in LAYERS:
        layer_label = LAYER_NAMES.get(layer, f"layer_{layer}")
        print(f"\n>>> Evaluating {layer_label} (index {layer}) <<<")

        topic_dfs = {}
        for topic in TOPICS:
            topic_dfs[topic] = load_topic_dataset(embeddings_dir, topic, model_tag, layer)

        layer_topic_accuracies = {}
        layer_topic_aucs = {}

        for test_idx, test_topic in enumerate(TOPICS):
            test_df = topic_dfs[test_topic]
            train_dfs = [topic_dfs[t] for t in TOPICS if t != test_topic]
            train_df = pd.concat(train_dfs, ignore_index=True)

            train_X = np.array([parse_embedding_str(e) for e in train_df['embeddings']])
            train_y = np.array(train_df['label'], dtype=int)
            test_X = np.array([parse_embedding_str(e) for e in test_df['embeddings']])
            test_y = np.array(test_df['label'], dtype=int)

            run_accuracies = []
            run_aucs = []

            for r in range(repeat_each):
                model = build_saplma_classifier(input_dim=train_X.shape[1])
                model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

                test_preds = model.predict(test_X, verbose=0).flatten()
                
                # Standard decision threshold = 0.5 as reported in Paper Tables 1 & 2
                binary_preds = (test_preds > 0.5).astype(int)
                acc = accuracy_score(test_y, binary_preds)
                run_accuracies.append(acc)

                fpr, tpr, _ = roc_curve(test_y, test_preds)
                run_aucs.append(auc(fpr, tpr))

            mean_acc = np.mean(run_accuracies)
            mean_auc = np.mean(run_aucs)
            layer_topic_accuracies[test_topic] = mean_acc
            layer_topic_aucs[test_topic] = mean_auc

            print(f"  [{test_topic:<12}] Acc = {mean_acc:.4f} (runs: {[round(x, 4) for x in run_accuracies]}), AUC = {mean_auc:.4f}")

        avg_acc = np.mean([layer_topic_accuracies[t] for t in TOPICS])
        avg_auc = np.mean([layer_topic_aucs[t] for t in TOPICS])
        print(f"  --> {layer_label} Average Accuracy across all 6 topics: {avg_acc:.4f} (Avg AUC: {avg_auc:.4f})")

        row = {
            "Layer": layer_label,
            "Layer_Index": layer,
            "Cities": layer_topic_accuracies["cities"],
            "Inventions": layer_topic_accuracies["inventions"],
            "Chemical_Elements": layer_topic_accuracies["elements"],
            "Animals": layer_topic_accuracies["animals"],
            "Companies": layer_topic_accuracies["companies"],
            "Scientific_Facts": layer_topic_accuracies["facts"],
            "Average_Accuracy": avg_acc,
            "Average_AUC": avg_auc
        }
        summary_rows.append(row)

    results_df = pd.DataFrame(summary_rows)
    out_file = os.path.join(output_dir, f"saplma_cross_topic_{model_tag}_results.csv")
    results_df.to_csv(out_file, index=False)
    print(f"\nFull results saved to: {out_file}\n")
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAPLMA cross-topic evaluation.")
    parser.add_argument("--model", type=str, default="opt", choices=["opt", "llama3", "llama2"])
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--rep", type=int, default=3, help="Number of repetitions with different random seeds (paper specifies 3).")
    args = parser.parse_args()

    run_cross_topic_evaluation(
        model_name=args.model,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        repeat_each=args.rep
    )
