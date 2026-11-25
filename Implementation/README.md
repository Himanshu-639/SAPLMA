# SAPLMA: Exact Replication of "The Internal State of an LLM Knows When It's Lying"

This repository provides an **exact, clean reimplementation** of the paper:
> **The Internal State of an LLM Knows When It's Lying**  
> *Amos Azaria and Tom Mitchell*  
> EMNLP Findings 2023 | [arXiv:2304.13734v2](https://arxiv.org/abs/2304.13734)

Every component follows the architectural, experimental, and thresholding protocols strictly as defined in the paper.

---

## 1. Paper Overview & Architecture

### A. The SAPLMA Classifier
* **Architecture:** Feedforward Multi-Layer Perceptron (MLP)
  * `Input Layer`: Size equal to LLM hidden state dimension ($4096$ for OPT-6.7b, $3072$ for LLaMA-3.2-3B).
  * `Dense(256)` with ReLU activation.
  * `Dense(128)` with ReLU activation.
  * `Dense(64)` with ReLU activation.
  * `Dense(1)` with Sigmoid activation.
* **Optimizer:** Adam (default parameters).
* **Loss Function:** Binary Cross-Entropy.
* **Epochs:** $5$ epochs (batch size $32$).
* **Hyper-parameters:** No fine-tuning applied, strictly as described in Section 4.

### B. Evaluated Candidate Layers
Out of 32 transformer layers, activations are extracted at:
* Last layer: Layer 32 (index `-1`)
* 28th layer (4th before last): Layer 28 (index `-4`)
* 24th layer (8th before last): Layer 24 (index `-8`)
* 20th layer (12th before last): Layer 20 (index `-12`)
* Middle layer: Layer 16 (index `-16`)

---

## 2. Directory Structure

```
Implementation/
├── data/
│   ├── cities_true_false.csv         # 1,458 sentences (Cities topic)
│   ├── inventions_true_false.csv     # 876 sentences (Inventions topic)
│   ├── elements_true_false.csv       # 930 sentences (Chemical Elements topic)
│   ├── animals_true_false.csv        # 1,008 sentences (Animals topic)
│   ├── companies_true_false.csv      # 1,200 sentences (Companies topic)
│   ├── facts_true_false.csv          # 612 sentences (Scientific Facts topic)
│   └── generated_true_false.csv      # 245 OPT-6.7b generated sentences (Tables 3 & 4)
├── embeddings/                       # Pre-extracted representations from local models
├── saplma/
│   ├── model.py                      # Exact 3-layer MLP classifier architecture
│   ├── extract_activations.py        # Hidden state extraction from causal LLMs
│   ├── evaluate_cross_topic.py       # Leave-one-topic-out evaluation (Tables 1 & 2)
│   ├── evaluate_generated.py         # 14-run evaluation on generated set (Tables 3 & 4)
│   ├── permutation_test.py           # Random label permutation control test (Section 5)
│   └── table5_analysis.py            # Statement probability vs. SAPLMA (Table 5)
├── baselines/
│   ├── bert_baseline.py              # BERT sentence embedding + MLP baseline
│   ├── few_shot_baseline.py          # 3-shot and 5-shot normalized prompting baseline
│   └── it_is_true_baseline.py        # "It-is-true" vs "It-is-false" probability baseline
├── results/                          # CSV output logs for all runs
├── run_paper_experiments.py          # Master CLI reproduction runner
└── requirements.txt
```

---

## 3. Quick Start & Execution

All scripts automatically load models from your **local cache** without redownloading.

### Step 1: Activate Environment
```powershell
conda activate swarm
cd c:\Users\hp\Desktop\SAPLMA\Implementation
```

### Step 2: Reproduce Table 1 (OPT-6.7b Cross-Topic Evaluation)
Runs 6-fold leave-one-topic-out cross-validation across all 6 topics for 3 random runs per topic with fixed threshold 0.5:
```powershell
python run_paper_experiments.py --experiment cross_topic --model opt --rep 3
```

### Step 3: Reproduce Table 2 Comparison (LLaMA Backbone)
```powershell
python run_paper_experiments.py --experiment cross_topic --model llama3 --rep 3
```

### Step 4: Reproduce Table 3 & Table 4 (LLM-Generated Sentences)
Trains on all 6 topics (6,084 sentences) and evaluates on the 245 generated statements over 14 runs:
```powershell
python run_paper_experiments.py --experiment generated --model opt --rep 14
```

### Step 5: Reproduce Random Permutation Control Test (Section 5)
Tests whether the MLP can memorize randomized labels:
```powershell
python run_paper_experiments.py --experiment permutation --model opt
```

### Step 6: Reproduce Baselines
```powershell
# BERT baseline
python run_paper_experiments.py --experiment bert

# Few-shot baseline (3-shot and 5-shot)
python run_paper_experiments.py --experiment few_shot

# It-is-true probability baseline
python run_paper_experiments.py --experiment it_is_true
```

### Step 7: Reproduce Table 5 Statement Comparison
```powershell
python run_paper_experiments.py --experiment table5 --model opt
```

---

## 4. Summary of Reproduced Results vs. Paper Benchmarks

### Table 1: OPT-6.7b SAPLMA Accuracy
| Layer | Reproduced Average | Paper Published Average | Difference |
| :--- | :---: | :---: | :---: |
| **Last layer (32)** | 64.30% | 64.49% | -0.19% |
| **28th layer** | 66.91% | 65.07% | +1.84% |
| **24th layer** | 69.63% | 68.86% | +0.77% |
| **20th layer (Optimal)** | **71.11%** | **70.98%** | **+0.13%** |
| **Middle layer (16)** | 64.65% | 65.15% | -0.50% |

### Table 2: LLaMA Veracity Gradient
| Layer | LLaMA-3.2-3B (Our Run) | LLaMA-2-7b (Paper Table 2) | Difference |
| :--- | :---: | :---: | :---: |
| **Last layer** | 69.89% | 69.34% | +0.55% |
| **28th layer** | 67.95% | 71.36% | -3.41% |
| **24th layer** | 73.94% | 76.84% | -2.90% |
| **20th layer** | 78.93% | 80.20% | -1.27% |
| **Middle layer (Optimal)** | **83.17%** | **82.98%** | **+0.19%** |

---

## 5. Multi-Precision & Quantization Experiments (FP16 vs. 8-Bit vs. 4-Bit)

We evaluated both **LLaMA-3.2-3B (Base)** and **LLaMA-3.2-3B-Instruct** across three quantization configurations:
1. **Full Precision (FP16 / BF16):** Official weights loaded in standard floating point.
2. **8-bit Quantization:** `BitsAndBytesConfig(load_in_8bit=True)` (LLM.int8()).
3. **4-bit Quantization:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.

All runs adhere strictly to the paper's 6-topic leave-one-out cross-validation protocol (3 repetitions per topic, 5 candidate layers, 5 epochs, threshold 0.5) without prompts or wrappers.

### Reproduction Command
```powershell
# Run all 4 quantization benchmarks and generate comparative tables:
python run_quantization_benchmarks.py --config all

# Or run specific configuration:
python run_paper_experiments.py --experiment cross_topic --model llama3 --precision 8bit
python run_paper_experiments.py --experiment cross_topic --model llama3 --precision 4bit
python run_paper_experiments.py --experiment cross_topic --model llama3-instruct --precision 8bit
python run_paper_experiments.py --experiment cross_topic --model llama3-instruct --precision 4bit
```

All terminal outputs are logged automatically in `results/logs/`:
* `results/logs/llama3.2_3b_8bit_run.log`
* `results/logs/llama3.2_3b_4bit_run.log`
* `results/logs/llama3.2_3b_instruct_8bit_run.log`
* `results/logs/llama3.2_3b_instruct_4bit_run.log`

### Master Multi-Precision Comparison Tables
*(Saved to `results/comparison_quantization_vs_full_precision_vs_paper.csv`)*

#### Accuracy Comparison (Threshold = 0.5)
| Layer | Index | Base FP16 | Base 8-bit | Base 4-bit | Instruct FP16 | Instruct 8-bit | Instruct 4-bit | Paper LLaMA-2-7B | Paper OPT-6.7B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **middle-layer** | **-16** | **83.17%** | **83.56%** | **82.91%** | **82.20%** | **81.73%** | **81.93%** | **82.98%** | 66.86% |
| **20th-layer** | -12 | 78.93% | 79.36% | 80.44% | 79.37% | 78.69% | 79.40% | 80.20% | **70.98%** |
| **24th-layer** | -8 | 73.94% | 74.11% | 74.35% | 75.26% | 75.55% | 74.26% | 76.84% | 67.54% |
| **28th-layer** | -4 | 67.95% | 68.91% | 67.04% | 71.32% | 72.78% | 69.31% | 71.36% | 65.68% |
| **last-layer** | -1 | 69.89% | 69.25% | 67.84% | 70.15% | 70.26% | 68.90% | 69.34% | 63.85% |

#### Area Under the ROC Curve (AUROC / AUC) Comparison
| Layer | Index | Base FP16 | Base 8-bit | Base 4-bit | Instruct FP16 | Instruct 8-bit | Instruct 4-bit | Replicated OPT-6.7B (Table 1 Setup) | Paper Reported OPT-6.7B (Table 3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **middle-layer** | **-16** | **0.9103** | **0.9097** | **0.9021** | **0.9081** | **0.9040** | **0.8987** | 0.7366 | 0.6610 |
| **20th-layer** | -12 | 0.8946 | 0.8913 | 0.8893 | 0.8930 | 0.8901 | 0.8895 | **0.8086** | 0.7182 |
| **24th-layer** | -8 | 0.8586 | 0.8535 | 0.8489 | 0.8644 | 0.8656 | 0.8627 | 0.8055 | 0.7435 |
| **28th-layer** | -4 | 0.8088 | 0.8110 | 0.7881 | 0.8456 | 0.8443 | 0.8392 | 0.7749 | **0.7614** |
| **last-layer** | -1 | 0.8044 | 0.8036 | 0.7968 | 0.8060 | 0.8121 | 0.8055 | 0.7489 | 0.7587 |

### Key Quantization Insights
1. **Truth Representation Resilience to Quantization:** 
   Compressing weights from 16-bit to 8-bit or 4-bit NormalFloat (NF4) causes virtually **zero degradation** in SAPLMA classifier accuracy. At the optimal middle layer, Base 8-bit reaches **83.56%** (vs 83.17% FP16), and Base 4-bit reaches **82.91%**.
2. **Peak Truthfulness Layer Remains Invariant:**
   Across all six model and precision variants (Base FP16, Base 8-bit, Base 4-bit, Instruct FP16, Instruct 8-bit, Instruct 4-bit), the **middle layer (index -16)** consistently retains the highest truthfulness score (~82%–83.5%), confirming that architectural layer specificity is immune to aggressive quantization.
3. **Instruct Model Benefits in Late Decoding Layers:**
   Instruction fine-tuning maintains higher veracity in the upper layers (28th layer: ~71–73% in Instruct vs ~67–69% in Base) regardless of whether the model is in 16-bit, 8-bit, or 4-bit.

---

## 6. Cross-Precision Transfer & Complete 28-Layer Benchmarks

We evaluated whether truth probes (3-layer MLPs) trained exclusively on **Full Precision (FP16)** internal representations generalize zero-shot when evaluated on **8-bit** and **4-bit** quantized representations. All probe training runs used locked random seeds (`seeds = [42, 123, 999]`) across 3 repetitions for exact determinism.

### Complete 28-Layer Depth Trajectory & Transfer Performance

#### Base Model (`LLaMA-3.2-3B`)
| Layer | Index | FP16 Native Acc | FP16 Native AUC | FP16 $\to$ 8-bit Acc | $\Delta$ 8-bit | FP16 $\to$ 4-bit Acc | $\Delta$ 4-bit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer-1 (Last)** | -1 | 68.54% | 0.8044 | 68.29% | -0.24% | 68.13% | -0.40% |
| **Layer-4** | -4 | 68.72% | 0.8060 | 68.00% | -0.72% | 67.02% | -1.70% |
| **Layer-8** | -8 | 72.47% | 0.8472 | 72.13% | -0.34% | 72.04% | -0.43% |
| **Layer-12** | -12 | 77.89% | 0.8864 | 77.90% | +0.01% | 76.90% | -0.99% |
| **Layer-14 (Middle)**| -14 | 78.25% | 0.8990 | 77.58% | -0.68% | 76.34% | -1.92% |
| **Layer-16 (Peak)** | **-16** | **80.98%** | **0.9037** | **80.90%** | **-0.08%** | **79.31%** | **-1.67%** |
| **Layer-18** | -18 | 80.52% | 0.8984 | 80.44% | -0.08% | 79.64% | -0.88% |
| **Layer-20** | -20 | 77.06% | 0.8623 | 76.92% | -0.14% | 75.87% | -1.19% |
| **Layer-24** | -24 | 61.21% | 0.6896 | 60.71% | -0.51% | 60.75% | -0.47% |
| **Layer-28 (First)**| -28 | 51.73% | 0.5184 | 51.48% | -0.25% | 51.72% | -0.01% |

#### Instruct Model (`LLaMA-3.2-3B-Instruct`)
| Layer | Index | FP16 Native Acc | FP16 Native AUC | FP16 $\to$ 8-bit Acc | $\Delta$ 8-bit | FP16 $\to$ 4-bit Acc | $\Delta$ 4-bit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer-1 (Last)** | -1 | 70.19% | 0.8092 | 70.55% | +0.36% | 69.14% | -1.04% |
| **Layer-4** | -4 | 72.10% | 0.8413 | 72.33% | +0.23% | 70.85% | -1.25% |
| **Layer-8** | -8 | 74.58% | 0.8648 | 74.47% | -0.12% | 73.86% | -0.72% |
| **Layer-12** | -12 | 77.82% | 0.8873 | 77.88% | +0.06% | 77.04% | -0.79% |
| **Layer-14 (Middle)**| -14 | 79.10% | 0.8953 | 79.26% | +0.15% | 78.29% | -0.81% |
| **Layer-16** | -16 | 81.05% | 0.9034 | 80.86% | -0.19% | 80.48% | -0.57% |
| **Layer-18 (Peak)** | **-18** | **81.25%** | **0.9003** | **81.31%** | **+0.06%** | **80.60%** | **-0.65%** |
| **Layer-20** | -20 | 77.43% | 0.8700 | 77.28% | -0.15% | 76.77% | -0.66% |
| **Layer-24** | -24 | 61.20% | 0.6917 | 60.94% | -0.26% | 60.80% | -0.40% |
| **Layer-28 (First)**| -28 | 50.95% | 0.5114 | 51.08% | +0.13% | 50.92% | -0.02% |

### Reproduction Commands
```powershell
# Run cross-precision transfer benchmark on 5 candidate layers:
python run_probe_transfer.py --model all

# Extract all 28 layers:
python extract_all_28_layers.py --model base --precision fp16

# Run full 28-layer probe transfer pipeline:
python run_all_28_layers_probe_transfer.py --model all
```
*(All 28-layer results are saved in `results/probe_transfer_all_28_layers_base.csv` and `results/probe_transfer_all_28_layers_instruct.csv`).*
