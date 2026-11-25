"""
Probe Cross-Precision Transfer Benchmark for SAPLMA
Tests whether truth probes (3-layer MLP) trained on Full Precision (FP16/BF16)
internal states generalize zero-shot to quantized (8-bit and 4-bit) internal states.

Evaluates both:
1. LLaMA-3.2-3B (Base)
2. LLaMA-3.2-3B-Instruct

Does NOT overwrite any existing result files or logs.
Saves all trained probe models into Implementation/saved_probes/ for future use.
Logs terminal outputs to Implementation/results/logs/ and Implementation/result/log/.
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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

class DualLogger:
    """Writes output to both stdout and a log file."""
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

def load_topic_dataset(embeddings_dir: str, topic: str, model_tag: str, layer: int) -> pd.DataFrame:
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

    raise FileNotFoundError(f"Could not find embeddings for topic '{topic}' (model_tag={model_tag}, layer={layer}) in '{embeddings_dir}'.")

def run_probe_transfer_for_model(
    model_type: str = "base", # 'base' or 'instruct'
    embeddings_dir: str = "embeddings",
    results_dir: str = "results",
    probes_dir: str = "saved_probes",
    epochs: int = 5,
    repeat_each: int = 3,
    batch_size: int = 32
):
    if model_type == "instruct":
        fp16_tag = "llama3.2_3b_instruct"
        bit8_tag = "llama3.2_3b_instruct_8bit"
        bit4_tag = "llama3.2_3b_instruct_4bit"
        model_display_name = "LLaMA-3.2-3B-Instruct"
    else:
        fp16_tag = "llama3.2_3b"
        bit8_tag = "llama3.2_3b_8bit"
        bit4_tag = "llama3.2_3b_4bit"
        model_display_name = "LLaMA-3.2-3B (Base)"

    model_probes_dir = os.path.join(probes_dir, fp16_tag)
    os.makedirs(model_probes_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" PROBE CROSS-PRECISION TRANSFER EVALUATION: {model_display_name}")
    print(f" Train: Full Precision (FP16) Probes -> Test on: FP16 (Native), 8-bit, and 4-bit")
    print(f" Saved Probes Destination: {model_probes_dir}")
    print("=" * 80 + "\n")

    summary_fp16 = []
    summary_to_8bit = []
    summary_to_4bit = []

    for layer in LAYERS:
        layer_label = LAYER_NAMES.get(layer, f"layer_{layer}")
        print(f"\n================================================================================")
        print(f">>> Evaluating Layer: {layer_label} (Index {layer})")
        print(f"================================================================================")

        # Load FP16 datasets (for training probe and native testing)
        fp16_dfs = {t: load_topic_dataset(embeddings_dir, t, fp16_tag, layer) for t in TOPICS}
        # Load 8-bit and 4-bit datasets (for transfer testing)
        bit8_dfs = {t: load_topic_dataset(embeddings_dir, t, bit8_tag, layer) for t in TOPICS}
        bit4_dfs = {t: load_topic_dataset(embeddings_dir, t, bit4_tag, layer) for t in TOPICS}

        fp16_accs, fp16_aucs = {}, {}
        to_8bit_accs, to_8bit_aucs = {}, {}
        to_4bit_accs, to_4bit_aucs = {}, {}

        for test_topic in TOPICS:
            print(f"\n--- [Topic: {test_topic}] Training FP16 Probe on remaining 5 topics ---")

            # Training set: remaining 5 topics in full precision
            train_dfs = [fp16_dfs[t] for t in TOPICS if t != test_topic]
            train_df = pd.concat(train_dfs, ignore_index=True)
            train_X = np.array([parse_embedding_str(e) for e in train_df['embeddings']])
            train_y = np.array(train_df['label'], dtype=int)

            # Test sets
            test_fp16_df = fp16_dfs[test_topic]
            test_fp16_X = np.array([parse_embedding_str(e) for e in test_fp16_df['embeddings']])
            test_fp16_y = np.array(test_fp16_df['label'], dtype=int)

            test_8bit_df = bit8_dfs[test_topic]
            test_8bit_X = np.array([parse_embedding_str(e) for e in test_8bit_df['embeddings']])
            test_8bit_y = np.array(test_8bit_df['label'], dtype=int)

            test_4bit_df = bit4_dfs[test_topic]
            test_4bit_X = np.array([parse_embedding_str(e) for e in test_4bit_df['embeddings']])
            test_4bit_y = np.array(test_4bit_df['label'], dtype=int)

            r_fp16_acc, r_fp16_auc = [], []
            r_8bit_acc, r_8bit_auc = [], []
            r_4bit_acc, r_4bit_auc = [], []

            for r in range(repeat_each):
                probe = build_saplma_classifier(input_dim=train_X.shape[1])
                probe.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

                # Save the trained full precision probe weights
                probe_save_path = os.path.join(
                    model_probes_dir,
                    f"probe_{fp16_tag}_layer{abs(layer)}_heldout_{test_topic}_rep{r}.weights.h5"
                )
                probe.save_weights(probe_save_path)

                # 1. Native FP16 evaluation
                p_fp16 = probe.predict(test_fp16_X, verbose=0).flatten()
                acc_fp16 = accuracy_score(test_fp16_y, (p_fp16 > 0.5).astype(int))
                fpr, tpr, _ = roc_curve(test_fp16_y, p_fp16)
                auc_fp16 = auc(fpr, tpr)
                r_fp16_acc.append(acc_fp16)
                r_fp16_auc.append(auc_fp16)

                # 2. Transfer evaluation on 8-bit quantized representations
                p_8bit = probe.predict(test_8bit_X, verbose=0).flatten()
                acc_8bit = accuracy_score(test_8bit_y, (p_8bit > 0.5).astype(int))
                fpr, tpr, _ = roc_curve(test_8bit_y, p_8bit)
                auc_8bit = auc(fpr, tpr)
                r_8bit_acc.append(acc_8bit)
                r_8bit_auc.append(auc_8bit)

                # 3. Transfer evaluation on 4-bit quantized representations
                p_4bit = probe.predict(test_4bit_X, verbose=0).flatten()
                acc_4bit = accuracy_score(test_4bit_y, (p_4bit > 0.5).astype(int))
                fpr, tpr, _ = roc_curve(test_4bit_y, p_4bit)
                auc_4bit = auc(fpr, tpr)
                r_4bit_acc.append(acc_4bit)
                r_4bit_auc.append(auc_4bit)

            fp16_accs[test_topic] = np.mean(r_fp16_acc)
            fp16_aucs[test_topic] = np.mean(r_fp16_auc)
            to_8bit_accs[test_topic] = np.mean(r_8bit_acc)
            to_8bit_aucs[test_topic] = np.mean(r_8bit_auc)
            to_4bit_accs[test_topic] = np.mean(r_4bit_acc)
            to_4bit_aucs[test_topic] = np.mean(r_4bit_auc)

            print(f"  [{test_topic:<12}] Native FP16  : Acc = {fp16_accs[test_topic]:.4f}, AUC = {fp16_aucs[test_topic]:.4f}")
            print(f"  [{test_topic:<12}] Transferred 8b: Acc = {to_8bit_accs[test_topic]:.4f}, AUC = {to_8bit_aucs[test_topic]:.4f} (Δ Acc: {to_8bit_accs[test_topic]-fp16_accs[test_topic]:+.4f})")
            print(f"  [{test_topic:<12}] Transferred 4b: Acc = {to_4bit_accs[test_topic]:.4f}, AUC = {to_4bit_aucs[test_topic]:.4f} (Δ Acc: {to_4bit_accs[test_topic]-fp16_accs[test_topic]:+.4f})")

        # Also train and save a Global Probe trained on ALL 6 topics for permanent deployment
        all_train_df = pd.concat([fp16_dfs[t] for t in TOPICS], ignore_index=True)
        all_train_X = np.array([parse_embedding_str(e) for e in all_train_df['embeddings']])
        all_train_y = np.array(all_train_df['label'], dtype=int)
        global_probe = build_saplma_classifier(input_dim=all_train_X.shape[1])
        global_probe.fit(all_train_X, all_train_y, epochs=epochs, batch_size=batch_size, verbose=0)
        global_save_path = os.path.join(model_probes_dir, f"probe_{fp16_tag}_layer{abs(layer)}_global_all_topics.weights.h5")
        global_probe.save_weights(global_save_path)
        print(f"  [Saved Global Probe]: {global_save_path}")

        # Averages for this layer
        avg_fp16_acc = np.mean([fp16_accs[t] for t in TOPICS])
        avg_fp16_auc = np.mean([fp16_aucs[t] for t in TOPICS])
        avg_8bit_acc = np.mean([to_8bit_accs[t] for t in TOPICS])
        avg_8bit_auc = np.mean([to_8bit_aucs[t] for t in TOPICS])
        avg_4bit_acc = np.mean([to_4bit_accs[t] for t in TOPICS])
        avg_4bit_auc = np.mean([to_4bit_aucs[t] for t in TOPICS])

        print(f"\n  >>> Layer Summary ({layer_label}) <<<")
        print(f"  Native FP16        : Mean Acc = {avg_fp16_acc*100:.2f}%, Mean AUC = {avg_fp16_auc:.4f}")
        print(f"  FP16 Probe -> 8-bit: Mean Acc = {avg_8bit_acc*100:.2f}%, Mean AUC = {avg_8bit_auc:.4f} (Δ Acc: {(avg_8bit_acc-avg_fp16_acc)*100:+.2f}%)")
        print(f"  FP16 Probe -> 4-bit: Mean Acc = {avg_4bit_acc*100:.2f}%, Mean AUC = {avg_4bit_auc:.4f} (Δ Acc: {(avg_4bit_acc-avg_fp16_acc)*100:+.2f}%)")

        row_fp16 = {
            "Layer": layer_label, "Layer_Index": layer,
            "Cities": fp16_accs["cities"], "Inventions": fp16_accs["inventions"],
            "Chemical_Elements": fp16_accs["elements"], "Animals": fp16_accs["animals"],
            "Companies": fp16_accs["companies"], "Scientific_Facts": fp16_accs["facts"],
            "Average_Accuracy": avg_fp16_acc, "Average_AUC": avg_fp16_auc
        }
        row_8bit = {
            "Layer": layer_label, "Layer_Index": layer,
            "Cities": to_8bit_accs["cities"], "Inventions": to_8bit_accs["inventions"],
            "Chemical_Elements": to_8bit_accs["elements"], "Animals": to_8bit_accs["animals"],
            "Companies": to_8bit_accs["companies"], "Scientific_Facts": to_8bit_accs["facts"],
            "Average_Accuracy": avg_8bit_acc, "Average_AUC": avg_8bit_auc
        }
        row_4bit = {
            "Layer": layer_label, "Layer_Index": layer,
            "Cities": to_4bit_accs["cities"], "Inventions": to_4bit_accs["inventions"],
            "Chemical_Elements": to_4bit_accs["elements"], "Animals": to_4bit_accs["animals"],
            "Companies": to_4bit_accs["companies"], "Scientific_Facts": to_4bit_accs["facts"],
            "Average_Accuracy": avg_4bit_acc, "Average_AUC": avg_4bit_auc
        }
        summary_fp16.append(row_fp16)
        summary_to_8bit.append(row_8bit)
        summary_to_4bit.append(row_4bit)

    df_fp16 = pd.DataFrame(summary_fp16)
    df_8bit = pd.DataFrame(summary_to_8bit)
    df_4bit = pd.DataFrame(summary_to_4bit)

    file_8bit = os.path.join(results_dir, f"probe_transfer_{model_type}_fp16_to_8bit_results.csv")
    file_4bit = os.path.join(results_dir, f"probe_transfer_{model_type}_fp16_to_4bit_results.csv")
    df_8bit.to_csv(file_8bit, index=False)
    df_4bit.to_csv(file_4bit, index=False)

    print(f"\n[Saved Results]: {file_8bit}")
    print(f"[Saved Results]: {file_4bit}\n")

    return df_fp16, df_8bit, df_4bit

def generate_transfer_master_comparison(results_dir="results"):
    """
    Creates a comprehensive comparative CSV contrasting:
    1. FP16 Native
    2. FP16 Probe -> 8-bit vs Native 8-bit
    3. FP16 Probe -> 4-bit vs Native 4-bit
    for both Base and Instruct models.
    """
    layers = ["middle-layer", "20th-layer", "24th-layer", "28th-layer", "last-layer"]
    layer_indices = {"middle-layer": -16, "20th-layer": -12, "24th-layer": -8, "28th-layer": -4, "last-layer": -1}

    # Helper to load and index by Layer
    def load_metric_map(filename):
        p = os.path.join(results_dir, filename)
        if not os.path.exists(p):
            return {}
        df = pd.read_csv(p)
        return {r["Layer"]: (r["Average_Accuracy"], r["Average_AUC"]) for _, r in df.iterrows()}

    base_fp16 = load_metric_map("saplma_cross_topic_llama3.2_3b_results.csv")
    base_native_8bit = load_metric_map("saplma_cross_topic_llama3.2_3b_8bit_results.csv")
    base_trans_8bit = load_metric_map("probe_transfer_base_fp16_to_8bit_results.csv")
    base_native_4bit = load_metric_map("saplma_cross_topic_llama3.2_3b_4bit_results.csv")
    base_trans_4bit = load_metric_map("probe_transfer_base_fp16_to_4bit_results.csv")

    inst_fp16 = load_metric_map("saplma_cross_topic_llama3.2_3b_instruct_results.csv")
    inst_native_8bit = load_metric_map("saplma_cross_topic_llama3.2_3b_instruct_8bit_results.csv")
    inst_trans_8bit = load_metric_map("probe_transfer_instruct_fp16_to_8bit_results.csv")
    inst_native_4bit = load_metric_map("saplma_cross_topic_llama3.2_3b_instruct_4bit_results.csv")
    inst_trans_4bit = load_metric_map("probe_transfer_instruct_fp16_to_4bit_results.csv")

    rows = []
    for l in layers:
        row = {
            "Layer": l,
            "Layer_Index": layer_indices[l],

            # Base Model
            "Base_FP16_Acc": f"{base_fp16.get(l, (0, 0))[0]*100:.2f}%",
            "Base_FP16_AUC": round(base_fp16.get(l, (0, 0))[1], 4),

            "Base_FP16_to_8bit_Acc": f"{base_trans_8bit.get(l, (0, 0))[0]*100:.2f}%",
            "Base_FP16_to_8bit_AUC": round(base_trans_8bit.get(l, (0, 0))[1], 4),
            "Base_Native_8bit_Acc": f"{base_native_8bit.get(l, (0, 0))[0]*100:.2f}%",
            "Base_Native_8bit_AUC": round(base_native_8bit.get(l, (0, 0))[1], 4),

            "Base_FP16_to_4bit_Acc": f"{base_trans_4bit.get(l, (0, 0))[0]*100:.2f}%",
            "Base_FP16_to_4bit_AUC": round(base_trans_4bit.get(l, (0, 0))[1], 4),
            "Base_Native_4bit_Acc": f"{base_native_4bit.get(l, (0, 0))[0]*100:.2f}%",
            "Base_Native_4bit_AUC": round(base_native_4bit.get(l, (0, 0))[1], 4),

            # Instruct Model
            "Instruct_FP16_Acc": f"{inst_fp16.get(l, (0, 0))[0]*100:.2f}%",
            "Instruct_FP16_AUC": round(inst_fp16.get(l, (0, 0))[1], 4),

            "Instruct_FP16_to_8bit_Acc": f"{inst_trans_8bit.get(l, (0, 0))[0]*100:.2f}%",
            "Instruct_FP16_to_8bit_AUC": round(inst_trans_8bit.get(l, (0, 0))[1], 4),
            "Instruct_Native_8bit_Acc": f"{inst_native_8bit.get(l, (0, 0))[0]*100:.2f}%",
            "Instruct_Native_8bit_AUC": round(inst_native_8bit.get(l, (0, 0))[1], 4),

            "Instruct_FP16_to_4bit_Acc": f"{inst_trans_4bit.get(l, (0, 0))[0]*100:.2f}%",
            "Instruct_FP16_to_4bit_AUC": round(inst_trans_4bit.get(l, (0, 0))[1], 4),
            "Instruct_Native_4bit_Acc": f"{inst_native_4bit.get(l, (0, 0))[0]*100:.2f}%",
            "Instruct_Native_4bit_AUC": round(inst_native_4bit.get(l, (0, 0))[1], 4),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    master_path = os.path.join(results_dir, "comparison_probe_transfer_fp16_to_quantized.csv")
    out_df.to_csv(master_path, index=False)
    print(f"\n[Master Cross-Precision Transfer Comparison Saved]: {master_path}")
    print(out_df.to_string(index=False))
    return out_df

def copy_logs_to_result_folder(source_log_file):
    """Ensures logs are mirrored into Implementation/result/log/ as well."""
    alt_log_dir = os.path.join(SCRIPT_DIR, "result", "log")
    os.makedirs(alt_log_dir, exist_ok=True)
    basename = os.path.basename(source_log_file)
    shutil.copy2(source_log_file, os.path.join(alt_log_dir, basename))
    print(f"[Log Mirrored to]: {os.path.join(alt_log_dir, basename)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Precision Transfer of SAPLMA Probes")
    parser.add_argument("--model", type=str, default="all", choices=["base", "instruct", "all"])
    args = parser.parse_args()

    models_to_run = ["base", "instruct"] if args.model == "all" else [args.model]

    for m in models_to_run:
        log_path = os.path.join(SCRIPT_DIR, "results", "logs", f"probe_transfer_{m}_fp16_to_quant.log")
        logger = DualLogger(log_path)
        sys.stdout = logger

        try:
            run_probe_transfer_for_model(
                model_type=m,
                embeddings_dir=os.path.join(SCRIPT_DIR, "embeddings"),
                results_dir=os.path.join(SCRIPT_DIR, "results"),
                probes_dir=os.path.join(SCRIPT_DIR, "saved_probes"),
                epochs=5,
                repeat_each=3,
                batch_size=32
            )
        finally:
            logger.close()
            sys.stdout = logger.terminal
            copy_logs_to_result_folder(log_path)

    # Generate master comparison table
    generate_transfer_master_comparison(results_dir=os.path.join(SCRIPT_DIR, "results"))

if __name__ == "__main__":
    main()
