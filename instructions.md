# Instructions to Run SAPLMA Replication & Benchmarks

This repository contains the exact replication of:
> **The Internal State of an LLM Knows When It's Lying** (Azaria & Mitchell, EMNLP Findings 2023)  
> Alongside extended benchmarks evaluating **Instruction Tuning** (`Llama-3.2-3B-Instruct`) and **Quantization** (8-bit and 4-bit precision).

---

## 1. Environment Setup

### Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA support (tested on NVIDIA RTX 2000 Ada with 16 GB VRAM)
* Conda or virtual environment

### Installation
```bash
# Clone the repository
git clone https://github.com/Himanshu-639/SAPLMA.git
cd SAPLMA

# Install dependencies
pip install -r requirements.txt
```

### Hugging Face Authentication
For downloading Meta LLaMA models, export your Hugging Face user access token:
```bash
# Linux/macOS
export HF_TOKEN="your_hf_token_here"

# Windows PowerShell
$env:HF_TOKEN="your_hf_token_here"
```
Or create a `.env` file in the root directory:
```env
HF_TOKEN = "your_hf_token_here"
```

---

## 2. Dataset Structure

All 7 datasets are located in `Implementation/data/`:
* `cities_true_false.csv` (1,458 statements)
* `inventions_true_false.csv` (876 statements)
* `elements_true_false.csv` (930 statements)
* `animals_true_false.csv` (1,008 statements)
* `companies_true_false.csv` (1,200 statements)
* `facts_true_false.csv` (612 statements)
* `generated_true_false.csv` (245 open-ended generated statements for Tables 3 & 4)

---

## 3. Reproduction Commands

Navigate to the `Implementation/` directory:
```bash
cd Implementation
```

### A. Core Paper Experiments (Table 1 & Table 2)
```bash
# 1. OPT-6.7b 6-topic leave-one-out cross-validation (Table 1)
python run_paper_experiments.py --experiment cross_topic --model opt --rep 3

# 2. LLaMA-3.2-3B (Base) cross-topic evaluation (Table 2 comparison)
python run_paper_experiments.py --experiment cross_topic --model llama3 --rep 3

# 3. LLaMA-3.2-3B-Instruct cross-topic evaluation
python run_paper_experiments.py --experiment cross_topic --model llama3-instruct --rep 3
```

### B. Multi-Precision & Quantization Benchmarks (8-bit & 4-bit)
Evaluates both Base and Instruct models in 8-bit (`BitsAndBytesConfig(load_in_8bit=True)`) and 4-bit (`nf4` with `bfloat16` compute):
```bash
# Run all 4 quantization benchmarks and compile master table:
python run_quantization_benchmarks.py --config all

# Or run individual configurations:
python run_paper_experiments.py --experiment cross_topic --model llama3 --precision 8bit
python run_paper_experiments.py --experiment cross_topic --model llama3 --precision 4bit
python run_paper_experiments.py --experiment cross_topic --model llama3-instruct --precision 8bit
python run_paper_experiments.py --experiment cross_topic --model llama3-instruct --precision 4bit
```

### C. Baselines (Paper Section 5 & Table 1)
```bash
# BERT sentence embedding + 3-layer MLP baseline
python run_paper_experiments.py --experiment bert

# Few-shot prompting baseline (3-shot and 5-shot)
python run_paper_experiments.py --experiment few_shot

# Conditional sequence probability baseline ("It is true that X" vs "It is false that X")
python run_paper_experiments.py --experiment it_is_true
```

### D. Generated Statements Evaluation (Tables 3 & 4)
Evaluates on the 245 open-ended generated statements across 14 repetitions:
```bash
# Table 3 (Fixed 0.5 threshold) & Table 4 (30% held-out optimal threshold)
python run_paper_experiments.py --experiment generated --model opt --rep 14
```

### E. Section 5 & Table 5 Analysis
```bash
# Random label permutation control experiment (proves model does not memorize)
python run_paper_experiments.py --experiment permutation --model opt

# Sentence probability vs SAPLMA 28th-layer prediction (Table 5 analysis)
python run_paper_experiments.py --experiment table5 --model opt
```

### F. Cross-Precision Transfer Benchmarks (FP16 $\to$ 8-bit & 4-bit)
Evaluates whether probes trained on Full Precision (FP16) generalize to quantized activations:
```bash
# 5 candidate layers transfer benchmark:
python run_probe_transfer.py --model all

# Full 28-layer depth extraction and probe transfer benchmark (with locked seeds):
python run_full_28_layers_pipeline.py
```

---

## 4. Results & Logs

All outputs are saved in `Implementation/results/`:
* `probe_transfer_all_28_layers_base.csv` & `probe_transfer_all_28_layers_instruct.csv`: Full 28-layer cross-precision results.
* `comparison_probe_transfer_fp16_to_quantized.csv`: Master FP16 to quantized transfer comparison.
* `comparison_quantization_vs_full_precision_vs_paper.csv`: Master multi-precision accuracy & AUROC comparison.
* `comparison_instruct_vs_pretrained_vs_paper.csv`: Comparison between Base, Instruct, and published paper numbers.
* `saplma_cross_topic_6.7b_results.csv`: OPT-6.7b topic-by-topic cross-validation results.
* `saplma_cross_topic_llama3.2_3b_results.csv`: LLaMA-3.2-3B Base cross-validation results.
* `saplma_cross_topic_llama3.2_3b_instruct_results.csv`: LLaMA-3.2-3B Instruct cross-validation results.
* `table3_generated_fixed_threshold_6.7b.csv` & `table4_generated_optimal_threshold_6.7b.csv`: Generated statements evaluation.
* `table5_statement_comparison.csv`: Statement probability vs SAPLMA score breakdown.
* `logs/`: Terminal stdout logs for all benchmark executions.
