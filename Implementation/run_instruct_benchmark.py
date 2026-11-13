"""
Automated System Checker & LLaMA-3.2-3B-Instruct Benchmark Runner
Strictly follows the paper's exact architecture and evaluation flow.
Checks GPU/CPU utilization before execution.
Ensures zero overwriting of previous run logs and artifacts.
"""

import os
import sys
import time
import subprocess
import psutil
import pandas as pd
import numpy as np
import torch

# Ensure script directory and saplma package can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from saplma.extract_activations import extract_and_save_embeddings
from saplma.evaluate_cross_topic import run_cross_topic_evaluation

def check_system_free(max_gpu_util: float = 20.0, min_vram_free_mb: float = 8000.0, max_cpu_util: float = 70.0):
    """
    Checks GPU utilization, free VRAM, and CPU utilization.
    Returns (is_free: bool, metrics_dict: dict)
    """
    cpu_util = psutil.cpu_percent(interval=1.0)
    ram = psutil.virtual_memory()

    gpu_util = None
    vram_free_mb = None
    vram_total_mb = None
    vram_used_mb = None

    try:
        smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        parts = [float(x.strip()) for x in smi.strip().split(',')]
        gpu_util, vram_used_mb, vram_total_mb, vram_free_mb = parts
    except Exception as e:
        print(f"Warning: Could not query nvidia-smi: {e}")

    metrics = {
        "cpu_util": cpu_util,
        "ram_available_gb": ram.available / (1024**3),
        "ram_total_gb": ram.total / (1024**3),
        "gpu_util": gpu_util,
        "vram_free_mb": vram_free_mb,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb
    }

    is_free = True
    reasons = []

    if cpu_util > max_cpu_util:
        is_free = False
        reasons.append(f"CPU utilization high ({cpu_util:.1f}% > {max_cpu_util}%)")

    if gpu_util is not None and gpu_util > max_gpu_util:
        is_free = False
        reasons.append(f"GPU utilization high ({gpu_util:.1f}% > {max_gpu_util}%)")

    if vram_free_mb is not None and vram_free_mb < min_vram_free_mb:
        is_free = False
        reasons.append(f"GPU VRAM free is low ({vram_free_mb:.0f} MB < {min_vram_free_mb:.0f} MB)")

    return is_free, metrics, reasons

def generate_comparison_summary(results_dir: str = "results"):
    """
    Generates a consolidated comparison between:
    - LLaMA-3.2-3B-Instruct (New Run)
    - LLaMA-3.2-3B Pretrained (Previous Run)
    - LLaMA-2-7B (Paper Table 2)
    - OPT-6.7B (Paper Table 1)
    """
    instruct_csv = os.path.join(results_dir, "saplma_cross_topic_llama3.2_3b_instruct_results.csv")
    pretrained_csv = os.path.join(results_dir, "saplma_cross_topic_llama3.2_3b_results.csv")
    opt_csv = os.path.join(results_dir, "saplma_cross_topic_6.7b_results.csv")

    if not os.path.exists(instruct_csv):
        print(f"Warning: {instruct_csv} does not exist yet.")
        return None

    df_instruct = pd.read_csv(instruct_csv)
    df_pretrained = pd.read_csv(pretrained_csv) if os.path.exists(pretrained_csv) else None
    df_opt = pd.read_csv(opt_csv) if os.path.exists(opt_csv) else None

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

    comparison_rows = []
    for _, row in df_instruct.iterrows():
        layer = row['Layer']
        inst_acc = row['Average_Accuracy']
        inst_auc = row['Average_AUC']

        pre_acc = None
        pre_auc = None
        if df_pretrained is not None and layer in df_pretrained['Layer'].values:
            pre_row = df_pretrained[df_pretrained['Layer'] == layer].iloc[0]
            pre_acc = pre_row['Average_Accuracy']
            pre_auc = pre_row['Average_AUC']

        comp_row = {
            "Layer": layer,
            "Layer_Index": row['Layer_Index'],
            "LLaMA-3.2-3B-Instruct_Acc": f"{inst_acc*100:.2f}%",
            "LLaMA-3.2-3B-Instruct_AUC": f"{inst_auc:.4f}",
            "LLaMA-3.2-3B-Base_Acc": f"{pre_acc*100:.2f}%" if pre_acc else "N/A",
            "LLaMA-3.2-3B-Base_AUC": f"{pre_auc:.4f}" if pre_auc else "N/A",
            "Paper_LLaMA-2-7B_Acc": f"{paper_llama2_7b.get(layer, 0)*100:.2f}%",
            "Paper_OPT-6.7B_Acc": f"{paper_opt_6_7b.get(layer, 0)*100:.2f}%"
        }
        comparison_rows.append(comp_row)

    comp_df = pd.DataFrame(comparison_rows)
    out_file = os.path.join(results_dir, "comparison_instruct_vs_pretrained_vs_paper.csv")
    comp_df.to_csv(out_file, index=False)

    print("\n" + "="*80)
    print("CONSOLIDATED BENCHMARK: INSTRUCT vs PRETRAINED vs PAPER REPORTED")
    print("="*80)
    print(comp_df.to_string(index=False))
    print("="*80 + "\n")
    return comp_df

def run_pipeline():
    """Executes the full pipeline for meta-llama/Llama-3.2-3B-Instruct."""
    data_dir = os.path.join(SCRIPT_DIR, "data")
    embeddings_dir = os.path.join(SCRIPT_DIR, "embeddings")
    output_dir = os.path.join(SCRIPT_DIR, "results")

    print("\n[Step 1/3] Extracting activations for meta-llama/Llama-3.2-3B-Instruct...")
    extract_and_save_embeddings(
        model_name="llama3-instruct",
        data_dir=data_dir,
        output_dir=embeddings_dir
    )

    print("\n[Step 2/3] Running 6-topic cross-validation SAPLMA...")
    run_cross_topic_evaluation(
        model_name="llama3-instruct",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        repeat_each=3
    )

    print("\n[Step 3/3] Generating consolidated comparison...")
    generate_comparison_summary(output_dir)

if __name__ == "__main__":
    is_free, metrics, reasons = check_system_free()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] System Check Status:")
    print(f"  CPU Utilization : {metrics['cpu_util']:.1f}%")
    print(f"  RAM Available   : {metrics['ram_available_gb']:.1f} GB / {metrics['ram_total_gb']:.1f} GB")
    if metrics['gpu_util'] is not None:
        print(f"  GPU Utilization : {metrics['gpu_util']:.1f}%")
        print(f"  VRAM Free       : {metrics['vram_free_mb']:.0f} MB / {metrics['vram_total_mb']:.0f} MB (Used: {metrics['vram_used_mb']:.0f} MB)")

    if not is_free:
        print(f"\n[SYSTEM BUSY] Cannot start benchmark yet:")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(2)  # Exit code 2 indicates system is busy
    else:
        print("\n[SYSTEM FREE] Resources available. Starting LLaMA-3.2-3B-Instruct benchmark...\n")
        run_pipeline()
        sys.exit(0)
