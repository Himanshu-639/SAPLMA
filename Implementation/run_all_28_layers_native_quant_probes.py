"""
Native Quantized Probe Training and Comparison across All 28 Layers.
Trains 3-layer MLP probes directly on 8-bit and 4-bit quantized activations for each of the 28 layers.
Compares:
1. FP16 Native Probes
2. FP16 Probes transferred zero-shot to Quantized states
3. Native Retrained Probes on Quantized states (8-bit and 4-bit)

Uses locked random keys [42, 123, 999] for exact determinism across all 3 repetitions.
Logs terminal outputs to results/logs/ and result/log/.
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from saplma.model import build_saplma_classifier, parse_embedding_str

NUM_LAYERS = 28
ALL_LAYERS = [-i for i in range(1, NUM_LAYERS + 1)]
TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]
RUN_SEEDS = [42, 123, 999]

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def load_topic_dataset(embeddings_dir: str, topic: str, model_tag: str, layer_num: int) -> pd.DataFrame:
    candidate_names = [topic]
    if topic == "cities":
        candidate_names.append("capitals")
    elif topic == "capitals":
        candidate_names.append("cities")

    for name in candidate_names:
        p = os.path.join(embeddings_dir, f"embeddings_with_labels_{name}{model_tag}_{layer_num}_rmv_period.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
        p_no_rmv = os.path.join(embeddings_dir, f"embeddings_with_labels_{name}{model_tag}_{layer_num}.csv")
        if os.path.exists(p_no_rmv):
            return pd.read_csv(p_no_rmv)

    raise FileNotFoundError(f"Missing embeddings for topic '{topic}', tag={model_tag}, layer={layer_num} in '{embeddings_dir}'.")

def train_and_eval_native_quant_probes(
    model_type="base",       # "base" or "instruct"
    precision="8bit",        # "8bit" or "4bit"
    embeddings_dir="embeddings",
    results_dir="results",
    probes_dir="saved_probes_28layers",
    epochs=5,
    repeat_each=3,
    batch_size=32
):
    base_tag = "llama3.2_3b_instruct" if model_type == "instruct" else "llama3.2_3b"
    model_tag = f"{base_tag}_{precision}"
    model_display = f"{'LLaMA-3.2-3B-Instruct' if model_type == 'instruct' else 'LLaMA-3.2-3B (Base)'} [{precision} Native]"

    model_probes_dir = os.path.join(probes_dir, model_tag)
    os.makedirs(model_probes_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" TRAINING NATIVE PROBES ON {precision.upper()} ACTIVATIONS ACROSS ALL 28 LAYERS")
    print(f" Model: {model_display} | Tag: {model_tag}")
    print(f" Checkpoint Destination: {model_probes_dir}")
    print(f" Protocol: 6-fold Leave-One-Topic-Out, Reps={repeat_each}, Epochs={epochs}, Seeds={RUN_SEEDS}")
    print("=" * 80 + "\n")

    summary_rows = []

    for layer in ALL_LAYERS:
        layer_num = abs(layer)
        layer_name = f"Layer-{layer_num}"
        if layer == -1:
            layer_name = "Layer-1 (Last)"
        elif layer == -14:
            layer_name = "Layer-14 (Middle)"
        elif layer == -28:
            layer_name = "Layer-28 (First)"

        print(f"\n>>> [{model_tag}] Training Native Probes for {layer_name} (Index {layer}, file index {layer_num}) <<<")

        topic_dfs = {t: load_topic_dataset(embeddings_dir, t, model_tag, layer_num) for t in TOPICS}
        topic_accs, topic_aucs = {}, {}

        for test_topic in TOPICS:
            train_dfs = [topic_dfs[t] for t in TOPICS if t != test_topic]
            train_df = pd.concat(train_dfs, ignore_index=True)
            train_X = np.array([parse_embedding_str(e) for e in train_df['embeddings']])
            train_y = np.array(train_df['label'], dtype=int)

            test_df = topic_dfs[test_topic]
            test_X = np.array([parse_embedding_str(e) for e in test_df['embeddings']])
            test_y = np.array(test_df['label'], dtype=int)

            r_accs = []
            r_aucs = []

            for r in range(repeat_each):
                seed = RUN_SEEDS[r % len(RUN_SEEDS)]
                tf.keras.utils.set_random_seed(seed)

                probe = build_saplma_classifier(input_dim=train_X.shape[1])
                probe.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

                # Save trained probe checkpoint
                save_p = os.path.join(model_probes_dir, f"probe_{model_tag}_L{layer_num}_heldout_{test_topic}_r{r}.weights.h5")
                probe.save_weights(save_p)

                preds = probe.predict(test_X, verbose=0).flatten()
                acc = accuracy_score(test_y, (preds > 0.5).astype(int))
                fpr, tpr, _ = roc_curve(test_y, preds)
                r_accs.append(acc)
                r_aucs.append(auc(fpr, tpr))

            mean_acc = np.mean(r_accs)
            mean_auc = np.mean(r_aucs)
            topic_accs[test_topic] = mean_acc
            topic_aucs[test_topic] = mean_auc

        # Train and save a global all-topic probe
        tf.keras.utils.set_random_seed(42)
        all_train_df = pd.concat([topic_dfs[t] for t in TOPICS], ignore_index=True)
        all_train_X = np.array([parse_embedding_str(e) for e in all_train_df['embeddings']])
        all_train_y = np.array(all_train_df['label'], dtype=int)
        global_probe = build_saplma_classifier(input_dim=all_train_X.shape[1])
        global_probe.fit(all_train_X, all_train_y, epochs=epochs, batch_size=batch_size, verbose=0)
        global_save_path = os.path.join(model_probes_dir, f"probe_{model_tag}_L{layer_num}_global_all_topics.weights.h5")
        global_probe.save_weights(global_save_path)

        avg_acc = np.mean([topic_accs[t] for t in TOPICS])
        avg_auc = np.mean([topic_aucs[t] for t in TOPICS])

        print(f"  --> {layer_name}: Average Accuracy = {avg_acc*100:.2f}% | Average AUC = {avg_auc:.4f}")

        row = {
            "Layer": layer_name,
            "Layer_Index": layer,
            "Layer_Num": layer_num,
            "Cities": topic_accs["cities"],
            "Inventions": topic_accs["inventions"],
            "Chemical_Elements": topic_accs["elements"],
            "Animals": topic_accs["animals"],
            "Companies": topic_accs["companies"],
            "Scientific_Facts": topic_accs["facts"],
            "Average_Accuracy": avg_acc,
            "Average_AUC": avg_auc
        }
        summary_rows.append(row)

    df_out = pd.DataFrame(summary_rows)
    out_csv = os.path.join(results_dir, f"saplma_all_28_layers_{model_tag}_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[Saved Native Results]: {out_csv}\n")
    return df_out

def build_comparison_table(model_type="base", results_dir="results"):
    """
    Creates a direct side-by-side comparison CSV for all 28 layers:
    - FP16 Native
    - FP16 -> 8-bit Transfer vs 8-bit Retrained
    - FP16 -> 4-bit Transfer vs 4-bit Retrained
    """
    transfer_csv = os.path.join(results_dir, f"probe_transfer_all_28_layers_{model_type}.csv")
    base_tag = "llama3.2_3b_instruct" if model_type == "instruct" else "llama3.2_3b"
    bit8_csv = os.path.join(results_dir, f"saplma_all_28_layers_{base_tag}_8bit_results.csv")
    bit4_csv = os.path.join(results_dir, f"saplma_all_28_layers_{base_tag}_4bit_results.csv")

    if not (os.path.exists(transfer_csv) and os.path.exists(bit8_csv) and os.path.exists(bit4_csv)):
        print(f"Warning: Cannot build comparison for {model_type}; missing files.")
        return None

    df_trans = pd.read_csv(transfer_csv)
    df_8bit = pd.read_csv(bit8_csv).set_index("Layer_Num")
    df_4bit = pd.read_csv(bit4_csv).set_index("Layer_Num")

    rows = []
    for _, r in df_trans.iterrows():
        ln = int(r["Layer_Num"])
        layer_name = r["Layer"]
        layer_idx = r["Layer_Index"]

        fp16_acc = r["FP16_Native_Acc"]
        fp16_auc = float(r["FP16_Native_AUC"])

        trans_8b_acc = r["FP16_to_8bit_Acc"]
        trans_8b_auc = float(r["FP16_to_8bit_AUC"])

        retrain_8b_acc_val = df_8bit.loc[ln, "Average_Accuracy"]
        retrain_8b_auc_val = df_8bit.loc[ln, "Average_AUC"]
        retrain_8b_acc = f"{retrain_8b_acc_val * 100:.2f}%"
        retrain_8b_auc = round(retrain_8b_auc_val, 4)

        trans_8b_acc_val = float(trans_8b_acc.replace("%", "")) / 100.0
        delta_8b = f"{(trans_8b_acc_val - retrain_8b_acc_val) * 100:+.2f}%"

        trans_4b_acc = r["FP16_to_4bit_Acc"]
        trans_4b_auc = float(r["FP16_to_4bit_AUC"])

        retrain_4b_acc_val = df_4bit.loc[ln, "Average_Accuracy"]
        retrain_4b_auc_val = df_4bit.loc[ln, "Average_AUC"]
        retrain_4b_acc = f"{retrain_4b_acc_val * 100:.2f}%"
        retrain_4b_auc = round(retrain_4b_auc_val, 4)

        trans_4b_acc_val = float(trans_4b_acc.replace("%", "")) / 100.0
        delta_4b = f"{(trans_4b_acc_val - retrain_4b_acc_val) * 100:+.2f}%"

        rows.append({
            "Layer": layer_name,
            "Layer_Index": layer_idx,
            "Layer_Num": ln,
            "FP16_Native_Acc": fp16_acc,
            "FP16_Native_AUC": fp16_auc,
            "FP16_to_8bit_Transfer_Acc": trans_8b_acc,
            "FP16_to_8bit_Transfer_AUC": trans_8b_auc,
            "8bit_Retrained_Acc": retrain_8b_acc,
            "8bit_Retrained_AUC": retrain_8b_auc,
            "Delta_8bit_Transfer_vs_Retrained": delta_8b,
            "FP16_to_4bit_Transfer_Acc": trans_4b_acc,
            "FP16_to_4bit_Transfer_AUC": trans_4b_auc,
            "4bit_Retrained_Acc": retrain_4b_acc,
            "4bit_Retrained_AUC": retrain_4b_auc,
            "Delta_4bit_Transfer_vs_Retrained": delta_4b,
        })

    comp_df = pd.DataFrame(rows)
    out_comp_csv = os.path.join(results_dir, f"all_28_layers_native_vs_transfer_comparison_{model_type}.csv")
    comp_df.to_csv(out_comp_csv, index=False)
    print(f"\n[Master Comparison Saved]: {out_comp_csv}\n")
    print(comp_df.to_string(index=False))
    return comp_df

def copy_logs(source_log):
    alt_log_dir = os.path.join(SCRIPT_DIR, "result", "log")
    os.makedirs(alt_log_dir, exist_ok=True)
    shutil.copy2(source_log, os.path.join(alt_log_dir, os.path.basename(source_log)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["base", "instruct", "all"])
    parser.add_argument("--precision", type=str, default="all", choices=["8bit", "4bit", "all"])
    args = parser.parse_args()

    models = ["base", "instruct"] if args.model == "all" else [args.model]
    precisions = ["8bit", "4bit"] if args.precision == "all" else [args.precision]

    for m in models:
        log_path = os.path.join(SCRIPT_DIR, "results", "logs", f"native_quant_probes_all_28_layers_{m}.log")
        logger = DualLogger(log_path)
        sys.stdout = logger
        try:
            for p in precisions:
                train_and_eval_native_quant_probes(
                    model_type=m,
                    precision=p,
                    embeddings_dir=os.path.join(SCRIPT_DIR, "embeddings"),
                    results_dir=os.path.join(SCRIPT_DIR, "results"),
                    probes_dir=os.path.join(SCRIPT_DIR, "saved_probes_28layers"),
                    epochs=5,
                    repeat_each=3,
                    batch_size=32
                )
            build_comparison_table(model_type=m, results_dir=os.path.join(SCRIPT_DIR, "results"))
        finally:
            logger.close()
            sys.stdout = logger.terminal
            copy_logs(log_path)

if __name__ == "__main__":
    main()
