# Results: Comparison of Instruct vs. Pretrained vs. Paper Baselines

**Source File:** [`comparison_instruct_vs_pretrained_vs_paper.csv`](file:///c:/Users/hp/Desktop/SAPLMA/Implementation/results/comparison_instruct_vs_pretrained_vs_paper.csv)

---

## 📊 Benchmark Results Table

| Layer | Layer Index | LLaMA-3.2-3B-Instruct<br>(Accuracy) | LLaMA-3.2-3B-Instruct<br>(ROC-AUC) | LLaMA-3.2-3B-Base<br>(Accuracy) | LLaMA-3.2-3B-Base<br>(ROC-AUC) | Paper: LLaMA-2-7B<br>(Accuracy) | Paper: OPT-6.7B<br>(Accuracy) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **middle-layer** | `-16` | **82.20%** | **0.9081** | **83.17%** | **0.9103** | **82.98%** | 66.86% |
| **20th-layer** | `-12` | 79.37% | 0.8930 | 78.93% | 0.8946 | 80.20% | **70.98%** |
| **24th-layer** | `-8` | 75.26% | 0.8644 | 73.94% | 0.8586 | 76.84% | 67.54% |
| **28th-layer** | `-4` | 71.32% | 0.8456 | 67.95% | 0.8088 | 71.36% | 65.68% |
| **last-layer** | `-1` | 70.15% | 0.8060 | 69.89% | 0.8044 | 69.34% | 63.85% |

---

## 📋 Column Descriptions

| Column | Description |
| :--- | :--- |
| **`Layer`** | Layer designation (middle to last layer) |
| **`Layer_Index`** | Relative negative layer index in 32-layer transformer (`-16` = Layer 16, `-1` = Layer 32) |
| **`LLaMA-3.2-3B-Instruct_Acc`** | Truth classification accuracy for LLaMA-3.2-3B-Instruct |
| **`LLaMA-3.2-3B-Instruct_AUC`** | ROC Area Under the Curve (AUC) for LLaMA-3.2-3B-Instruct |
| **`LLaMA-3.2-3B-Base_Acc`** | Truth classification accuracy for pretrained LLaMA-3.2-3B |
| **`LLaMA-3.2-3B-Base_AUC`** | ROC Area Under the Curve (AUC) for pretrained LLaMA-3.2-3B |
| **`Paper_LLaMA-2-7B_Acc`** | Accuracy reported in Table 2 of Azaria & Mitchell (EMNLP Findings 2023) |
| **`Paper_OPT-6.7B_Acc`** | Accuracy reported in Table 1 of Azaria & Mitchell (EMNLP Findings 2023) |

---

## 🔍 Key Highlights

* **Peak Accuracy at Middle Layer (`-16`):** All LLaMA models reach their highest performance at the middle layer (~82%–83% accuracy, >0.90 AUC).
* **Instruct vs. Base:** Instruction tuning preserves the internal representation of truth while slightly improving deeper layer performance (e.g., +3.37% at layer `-4`).
* **Model Scale Efficiency:** LLaMA-3.2-3B matches or exceeds the larger LLaMA-2-7B across all probed layers and substantially outperforms OPT-6.7B.
