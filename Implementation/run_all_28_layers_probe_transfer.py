"""
Cross-Precision Probe Transfer across All 28 Hidden Layers of LLaMA-3.2-3B.
Trains 3-layer MLP truth probes on Full Precision (FP16) representations for each layer (1 to 28),
saves the trained models, and tests transferability to 8-bit and 4-bit quantized states.

Generates full 28-layer veracity gradient comparisons for Base and Instruct models.
Logs output to results/logs/ and result/log/.
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
# Layers 1 to 28 (where -1 is last layer, -14 is middle, -28 is first layer)
ALL_LAYERS = [-i for i in range(1, NUM_LAYERS + 1)]
TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]

# Locked random seeds for the 3 probe training repetitions to guarantee exact reproducibility
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

def run_probe_transfer_all_28_layers(
    model_type="base",
    embeddings_dir="embeddings",
    results_dir="results",
    probes_dir="saved_probes_28layers",
    epochs=5,
    repeat_each=3,
    batch_size=32
):
    if model_type == "instruct":
        fp16_tag = "llama3.2_3b_instruct"
        bit8_tag = "llama3.2_3b_instruct_8bit"
        bit4_tag = "llama3.2_3b_instruct_4bit"
        model_display = "LLaMA-3.2-3B-Instruct"
    else:
        fp16_tag = "llama3.2_3b"
        bit8_tag = "llama3.2_3b_8bit"
        bit4_tag = "llama3.2_3b_4bit"
        model_display = "LLaMA-3.2-3B (Base)"

    model_probes_dir = os.path.join(probes_dir, fp16_tag)
    os.makedirs(model_probes_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" SAPLMA 28-LAYER CROSS-PRECISION PROBE TRANSFER: {model_display}")
    print(f" Train on FP16 -> Zero-Shot Transfer Test on 8-bit and 4-bit")
    print(f" Destination for Saved Probes: {model_probes_dir}")
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

        print(f"\n>>> Evaluating {layer_name} (Index {layer}, file index {layer_num}) <<<")

        fp16_dfs = {t: load_topic_dataset(embeddings_dir, t, fp16_tag, layer_num) for t in TOPICS}
        bit8_dfs = {t: load_topic_dataset(embeddings_dir, t, bit8_tag, layer_num) for t in TOPICS}
        bit4_dfs = {t: load_topic_dataset(embeddings_dir, t, bit4_tag, layer_num) for t in TOPICS}

        fp16_accs, fp16_aucs = {}, {}
        to_8bit_accs, to_8bit_aucs = {}, {}
        to_4bit_accs, to_4bit_aucs = {}, {}

        for test_topic in TOPICS:
            # 5-topic training set in FP16
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
                seed = RUN_SEEDS[r % len(RUN_SEEDS)]
                tf.keras.utils.set_random_seed(seed)

                probe = build_saplma_classifier(input_dim=train_X.shape[1])
                probe.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

                # Save trained probe checkpoint
                save_p = os.path.join(model_probes_dir, f"probe_{fp16_tag}_L{layer_num}_heldout_{test_topic}_r{r}.weights.h5")
                probe.save_weights(save_p)

                # Predictions
                p_fp16 = probe.predict(test_fp16_X, verbose=0).flatten()
                r_fp16_acc.append(accuracy_score(test_fp16_y, (p_fp16 > 0.5).astype(int)))
                fpr, tpr, _ = roc_curve(test_fp16_y, p_fp16)
                r_fp16_auc.append(auc(fpr, tpr))

                p_8bit = probe.predict(test_8bit_X, verbose=0).flatten()
                r_8bit_acc.append(accuracy_score(test_8bit_y, (p_8bit > 0.5).astype(int)))
                fpr, tpr, _ = roc_curve(test_8bit_y, p_8bit)
                r_8bit_auc.append(auc(fpr, tpr))

                p_4bit = probe.predict(test_4bit_X, verbose=0).flatten()
                r_4bit_acc.append(accuracy_score(test_4bit_y, (p_4bit > 0.5).astype(int)))
                fpr, tpr, _ = roc_curve(test_4bit_y, p_4bit)
                r_4bit_auc.append(auc(fpr, tpr))

            fp16_accs[test_topic] = np.mean(r_fp16_acc)
            fp16_aucs[test_topic] = np.mean(r_fp16_auc)
            to_8bit_accs[test_topic] = np.mean(r_8bit_acc)
            to_8bit_aucs[test_topic] = np.mean(r_8bit_auc)
            to_4bit_accs[test_topic] = np.mean(r_4bit_acc)
            to_4bit_aucs[test_topic] = np.mean(r_4bit_auc)

        # Global probe on all 6 topics (locked random seed 42)
        tf.keras.utils.set_random_seed(42)
        all_train_df = pd.concat([fp16_dfs[t] for t in TOPICS], ignore_index=True)
        all_train_X = np.array([parse_embedding_str(e) for e in all_train_df['embeddings']])
        all_train_y = np.array(all_train_df['label'], dtype=int)
        global_probe = build_saplma_classifier(input_dim=all_train_X.shape[1])
        global_probe.fit(all_train_X, all_train_y, epochs=epochs, batch_size=batch_size, verbose=0)
        global_save_path = os.path.join(model_probes_dir, f"probe_{fp16_tag}_L{layer_num}_global_all_topics.weights.h5")
        global_probe.save_weights(global_save_path)

        avg_fp16_acc = np.mean([fp16_accs[t] for t in TOPICS])
        avg_fp16_auc = np.mean([fp16_aucs[t] for t in TOPICS])
        avg_8bit_acc = np.mean([to_8bit_accs[t] for t in TOPICS])
        avg_8bit_auc = np.mean([to_8bit_aucs[t] for t in TOPICS])
        avg_4bit_acc = np.mean([to_4bit_accs[t] for t in TOPICS])
        avg_4bit_auc = np.mean([to_4bit_aucs[t] for t in TOPICS])

        print(f"  --> {layer_name}: FP16 Acc={avg_fp16_acc*100:.2f}% (AUC={avg_fp16_auc:.4f}) | Trans 8b Acc={avg_8bit_acc*100:.2f}% (AUC={avg_8bit_auc:.4f}) | Trans 4b Acc={avg_4bit_acc*100:.2f}% (AUC={avg_4bit_auc:.4f})")

        row = {
            "Layer": layer_name,
            "Layer_Index": layer,
            "Layer_Num": layer_num,
            "FP16_Native_Acc": f"{avg_fp16_acc*100:.2f}%",
            "FP16_Native_AUC": round(avg_fp16_auc, 4),
            "FP16_to_8bit_Acc": f"{avg_8bit_acc*100:.2f}%",
            "FP16_to_8bit_AUC": round(avg_8bit_auc, 4),
            "Delta_8bit_Acc": f"{(avg_8bit_acc - avg_fp16_acc)*100:+.2f}%",
            "FP16_to_4bit_Acc": f"{avg_4bit_acc*100:.2f}%",
            "FP16_to_4bit_AUC": round(avg_4bit_auc, 4),
            "Delta_4bit_Acc": f"{(avg_4bit_acc - avg_fp16_acc)*100:+.2f}%"
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    out_file = os.path.join(results_dir, f"probe_transfer_all_28_layers_{model_type}.csv")
    df_summary.to_csv(out_file, index=False)
    print(f"\n[Saved 28-Layer Summary]: {out_file}\n")
    print(df_summary.to_string(index=False))
    return df_summary

def copy_logs(source_log):
    alt_log_dir = os.path.join(SCRIPT_DIR, "result", "log")
    os.makedirs(alt_log_dir, exist_ok=True)
    shutil.copy2(source_log, os.path.join(alt_log_dir, os.path.basename(source_log)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["base", "instruct", "all"])
    args = parser.parse_args()

    models = ["base", "instruct"] if args.model == "all" else [args.model]
    for m in models:
        log_path = os.path.join(SCRIPT_DIR, "results", "logs", f"probe_transfer_all_28_layers_{m}.log")
        logger = DualLogger(log_path)
        sys.stdout = logger
        try:
            run_probe_transfer_all_28_layers(
                model_type=m,
                embeddings_dir=os.path.join(SCRIPT_DIR, "embeddings"),
                results_dir=os.path.join(SCRIPT_DIR, "results"),
                probes_dir=os.path.join(SCRIPT_DIR, "saved_probes_28layers"),
                epochs=5,
                repeat_each=3,
                batch_size=32
            )
        finally:
            logger.close()
            sys.stdout = logger.terminal
            copy_logs(log_path)

if __name__ == "__main__":
    main()
