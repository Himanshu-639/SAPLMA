"""
Master Pipeline for 28-Layer Activation Extraction and Cross-Precision Probe Transfer.
Runs:
1. Activation Extraction for all 28 layers:
   - Base: FP16, 8-bit, 4-bit
   - Instruct: FP16, 8-bit, 4-bit
2. Probe Training (FP16), Probe Checkpointing, and Zero-shot Cross-Precision Testing:
   - Base (all 28 layers)
   - Instruct (all 28 layers)
3. Table compilation and logging.
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = r"C:\Users\hp\miniconda3\envs\swarm\python.exe"

def run_step(cmd_args, desc):
    print("\n" + "#" * 80)
    print(f" PIPELINE STEP: {desc}")
    print(f" COMMAND: {' '.join(cmd_args)}")
    print("#" * 80 + "\n")
    t0 = time.time()
    res = subprocess.run(cmd_args, cwd=SCRIPT_DIR)
    t1 = time.time()
    if res.returncode != 0:
        print(f"ERROR: Step failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED: {desc} in {(t1-t0)/60:.2f} minutes.")

def main():
    total_start = time.time()
    print("=" * 80)
    print(" STARTING COMPLETE 28-LAYER PIPELINE")
    print("=" * 80)

    # 1. Base FP16
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "base", "--precision", "fp16"], "Extract Base FP16 (28 Layers)")
    # 2. Base 8-bit
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "base", "--precision", "8bit"], "Extract Base 8-bit (28 Layers)")
    # 3. Base 4-bit
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "base", "--precision", "4bit"], "Extract Base 4-bit (28 Layers)")

    # 4. Instruct FP16
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "instruct", "--precision", "fp16"], "Extract Instruct FP16 (28 Layers)")
    # 5. Instruct 8-bit
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "instruct", "--precision", "8bit"], "Extract Instruct 8-bit (28 Layers)")
    # 6. Instruct 4-bit
    run_step([PYTHON_EXE, "-u", "extract_all_28_layers.py", "--model", "instruct", "--precision", "4bit"], "Extract Instruct 4-bit (28 Layers)")

    # 7. Cross-Precision Transfer Benchmark
    run_step([PYTHON_EXE, "-u", "run_all_28_layers_probe_transfer.py", "--model", "all"], "28-Layer Probe Training & Transfer Testing")

    total_time = (time.time() - total_start) / 60
    print("\n" + "=" * 80)
    print(f" ALL 28-LAYER BENCHMARKS COMPLETED IN {total_time:.2f} MINUTES!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
