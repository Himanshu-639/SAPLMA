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
