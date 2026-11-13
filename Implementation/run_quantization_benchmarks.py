"""
Master Quantization Benchmark Runner for SAPLMA
Evaluates LLaMA-3.2-3B (Base) and LLaMA-3.2-3B-Instruct in:
- 8-bit precision: BitsAndBytesConfig(load_in_8bit=True)
- 4-bit precision: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

Logs all terminal outputs to Implementation/results/logs/
Generates master comparative tables without overwriting any previous runs.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Ensure root directory is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from saplma.extract_activations import extract_and_save_embeddings
from saplma.evaluate_cross_topic import run_cross_topic_evaluation

class DualLogger:
    """Writes to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
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

def generate_multi_precision_table(output_dir="results"):
    """
    Assembles a master comparison table across all evaluated precisions:
    - Base: FP16, 8-bit, 4-bit
    - Instruct: FP16, 8-bit, 4-bit
    - Paper: LLaMA-2-7B (Table 2), OPT-6.7B (Table 1)
    """
    layers = ["middle-layer", "20th-layer", "24th-layer", "28th-layer", "last-layer"]
    layer_indices = {"middle-layer": -16, "20th-layer": -12, "24th-layer": -8, "28th-layer": -4, "last-layer": -1}

    # Load result dataframes if they exist
    results_map = {}
    csv_keys = {
        "Base_FP16": "saplma_cross_topic_llama3.2_3b_results.csv",
        "Base_8bit": "saplma_cross_topic_llama3.2_3b_8bit_results.csv",
        "Base_4bit": "saplma_cross_topic_llama3.2_3b_4bit_results.csv",
        "Instruct_FP16": "saplma_cross_topic_llama3.2_3b_instruct_results.csv",
        "Instruct_8bit": "saplma_cross_topic_llama3.2_3b_instruct_8bit_results.csv",
        "Instruct_4bit": "saplma_cross_topic_llama3.2_3b_instruct_4bit_results.csv",
        "OPT_6.7B_FP16": "saplma_cross_topic_6.7b_results.csv"
    }

    for key, filename in csv_keys.items():
        p = os.path.join(output_dir, filename)
        if os.path.exists(p):
            results_map[key] = pd.read_csv(p).set_index("Layer")

    # Paper reported numbers
    paper_llama2_7b = {
        "middle-layer": 0.8298,
        "20th-layer": 0.8020,
        "24th-layer": 0.7684,
        "28th-layer": 0.7136,
        "last-layer": 0.6934
    }
    paper_opt_6_7b = {
        "20th-layer": 0.7098,
        "24th-layer": 0.6754,
        "28th-layer": 0.6568,
        "middle-layer": 0.6686,
        "last-layer": 0.6385
    }

    rows = []
    for layer in layers:
        row = {
            "Layer": layer,
            "Layer_Index": layer_indices[layer]
        }

        # Base models
        for prec in ["FP16", "8bit", "4bit"]:
            k = f"Base_{prec}"
            if k in results_map and layer in results_map[k].index:
                acc = results_map[k].loc[layer, "Average_Accuracy"]
                auc = results_map[k].loc[layer, "Average_AUC"]
                row[f"Base_{prec}_Acc"] = f"{acc*100:.2f}%"
                row[f"Base_{prec}_AUC"] = f"{auc:.4f}"
            else:
                row[f"Base_{prec}_Acc"] = "N/A"
                row[f"Base_{prec}_AUC"] = "N/A"

        # Instruct models
        for prec in ["FP16", "8bit", "4bit"]:
            k = f"Instruct_{prec}"
            if k in results_map and layer in results_map[k].index:
                acc = results_map[k].loc[layer, "Average_Accuracy"]
                auc = results_map[k].loc[layer, "Average_AUC"]
                row[f"Instruct_{prec}_Acc"] = f"{acc*100:.2f}%"
                row[f"Instruct_{prec}_AUC"] = f"{auc:.4f}"
            else:
                row[f"Instruct_{prec}_Acc"] = "N/A"
                row[f"Instruct_{prec}_AUC"] = "N/A"

        row["Paper_LLaMA-2-7B_Acc"] = f"{paper_llama2_7b.get(layer, 0)*100:.2f}%"
        row["Paper_OPT-6.7B_Acc"] = f"{paper_opt_6_7b.get(layer, 0)*100:.2f}%"

        # AUCs from our replicated OPT-6.7B cross-topic run (Table 1 setup)
        if "OPT_6.7B_FP16" in results_map and layer in results_map["OPT_6.7B_FP16"].index:
            opt_auc = results_map["OPT_6.7B_FP16"].loc[layer, "Average_AUC"]
            row["OPT-6.7B_Replicated_AUC"] = f"{opt_auc:.4f}"
        else:
            row["OPT-6.7B_Replicated_AUC"] = "N/A"

        # Paper published AUCs from Table 3 (the only table where authors report AUC)
        paper_table3_auc = {
            "28th-layer": 0.7614,
            "last-layer": 0.7587,
            "24th-layer": 0.7435,
            "20th-layer": 0.7182,
            "middle-layer": 0.6610
        }
        row["Paper_OPT-6.7B_Table3_AUC"] = f"{paper_table3_auc.get(layer, 0):.4f}"
        rows.append(row)

    df_master = pd.DataFrame(rows)
    master_csv = os.path.join(output_dir, "comparison_quantization_vs_full_precision_vs_paper.csv")
    df_master.to_csv(master_csv, index=False)
    print(f"\nMaster multi-precision comparison table saved to: {master_csv}\n")
    print(df_master.to_string(index=False))
    return df_master

def run_benchmarks(target_config: str = "all"):
    data_dir = os.path.join(SCRIPT_DIR, "data")
    embeddings_dir = os.path.join(SCRIPT_DIR, "embeddings")
    output_dir = os.path.join(SCRIPT_DIR, "results")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    configs = [
        {"model": "llama3", "precision": "8bit", "tag": "llama3.2_3b_8bit", "title": "LLaMA-3.2-3B (Base) 8-bit"},
        {"model": "llama3", "precision": "4bit", "tag": "llama3.2_3b_4bit", "title": "LLaMA-3.2-3B (Base) 4-bit"},
        {"model": "llama3-instruct", "precision": "8bit", "tag": "llama3.2_3b_instruct_8bit", "title": "LLaMA-3.2-3B-Instruct 8-bit"},
        {"model": "llama3-instruct", "precision": "4bit", "tag": "llama3.2_3b_instruct_4bit", "title": "LLaMA-3.2-3B-Instruct 4-bit"},
    ]

    if target_config != "all":
        configs = [c for c in configs if target_config in c["tag"] or (c["model"] == target_config.split("_")[0] and c["precision"] in target_config)]

    original_stdout = sys.stdout

    for cfg in configs:
        model_name = cfg["model"]
        precision = cfg["precision"]
        tag = cfg["tag"]
        title = cfg["title"]

        result_file = os.path.join(output_dir, f"saplma_cross_topic_{tag}_results.csv")
        if os.path.exists(result_file):
            print(f"[SKIP] Results for {title} already exist at {result_file}. Skipping execution.")
            continue

        log_file = os.path.join(logs_dir, f"{tag}_run.log")
        logger = DualLogger(log_file)
        sys.stdout = logger

        try:
            print("\n" + "="*80)
            print(f" STARTING BENCHMARK: {title}")
            print(f" Model: {model_name} | Precision: {precision} | Tag: {tag}")
            print(f" Log File: {log_file}")
            print("="*80 + "\n")

            # 1. Activation Extraction
            print(f"[Step 1/2] Extracting activations across 6 topics for 5 layers...")
            extract_and_save_embeddings(
                model_name=model_name,
                precision=precision,
                data_dir=data_dir,
                output_dir=embeddings_dir
            )

            # 2. Cross-Topic Evaluation
            print(f"\n[Step 2/2] Running 6-topic leave-one-out cross-validation (Repetitions=3)...")
            run_cross_topic_evaluation(
                model_name=model_name,
                precision=precision,
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
                repeat_each=3
            )

            print(f"\n[COMPLETED] Successfully finished {title}!\n")
        except Exception as e:
            print(f"\n[ERROR] Failed during {title}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            logger.close()

    # Generate master comparison table
    print("\n" + "="*80)
    print(" COMPILING MASTER COMPARISON TABLE ACROSS ALL PRECISIONS")
    print("="*80)
    generate_multi_precision_table(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantization Benchmarks for SAPLMA.")
    parser.add_argument("--config", type=str, default="all", help="Target configuration or 'all'")
    args = parser.parse_args()
    run_benchmarks(args.config)
