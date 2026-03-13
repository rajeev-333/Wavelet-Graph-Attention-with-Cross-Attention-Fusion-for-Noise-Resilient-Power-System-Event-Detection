


#  Wavelet-GAT-MCA: Robust Power System Event Classification under Noisy PMU Conditions

> A M.Tech-level research implementation for realistic power system disturbance detection using Wavelet encoding, Graph-based spatial modeling, and Cross-Attention fusion.

---

##  Overview

Power systems are increasingly monitored using **Phasor Measurement Units (PMUs)**. However, real-world PMU signals are noisy and disturbances can be subtle, making event classification challenging.

This project proposes a novel hybrid architecture:

**Wavelet + Graph Attention + Multi-Head Cross Attention (Wavelet-GAT-MCA)**

The model is evaluated against:

* A CNN-LSTM style baseline
* Multiple ML benchmarks (KNN, SVM, DT, MLP)
* Multi-level noise testing (40 dB → 20 dB)

The results demonstrate **superior robustness and generalization under realistic noisy conditions.**

---

##  Proposed Architecture

```
PMU Time-Series Signals
        ↓
Wavelet + Gramian Angular Field Encoding
        ↓
64×64×3 Image Representation
        ↓
Graph Attention (IEEE-39 Bus Topology)
        ↓
Statistical Temporal Feature Extraction
        ↓
Multi-Head Cross-Attention Fusion
        ↓
Deep MLP Classifier
        ↓
Event Classification
```

---

## ⚙️ Key Features

* ✔ IEEE-39 Bus System Simulation
* ✔ Realistic Power System Event Modeling
* ✔ Multi-scale Continuous Wavelet Transform (CWT)
* ✔ Gramian Angular Field (GAF) Temporal Encoding
* ✔ Graph-based Spatial Feature Aggregation
* ✔ Cross-Attention Spatial-Temporal Fusion
* ✔ Noise Robustness Evaluation (20–40 dB)
* ✔ Benchmark Comparison with 7 Models
* ✔ Confusion Matrix & Performance Visualization

---

##  Event Classes

| Label  | Description          |
| ------ | -------------------- |
| Stable | Normal operation     |
| GT     | Generator Trip       |
| LD     | Load Disconnection   |
| AG     | Single Phase Fault   |
| AB     | Phase-to-Phase Fault |
| ABCG   | Three-Phase Fault    |

---

## 🔬 Challenging Experimental Setup

To ensure realistic evaluation:

* 🔹 Only **250 samples per event** (generalization testing)
* 🔹 Training data includes **25–35 dB noise**
* 🔹 Testing performed at multiple noise levels:

  * 40 dB
  * 35 dB
  * 30 dB
  * 25 dB
  * 20 dB
* 🔹 Reduced event magnitude to simulate subtle disturbances

---

## 📈 Performance Outputs

The project automatically generates:

* `model_comparison_table.csv`
* `noise_resistance_table.csv`
* `model_comparison.png`
* `noise_resistance_comparison.png`
* `confusion_matrix.png`

Example evaluation metrics:

* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1-Score



##  Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/rajeev-333/Wavelet-Graph-Attention-with-Cross-Attention-Fusion-for-Noise-Resilient-Power-System-Event-Detection
cd Wavelet-GAT-MCA: Robust Power System Event Classification under Noisy PMU Conditions
```

### 2️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy pywavelets
```

---

## ▶️ How to Run

```bash
python complex.py
```

The script will:

1. Generate noisy PMU datasets
2. Train baseline and proposed models
3. Train benchmark ML models
4. Evaluate multi-level noise robustness
5. Generate comparison tables and plots




##  Benchmark Models

* K-Nearest Neighbors
* Support Vector Machine (RBF)
* Decision Tree
* Multi-Layer Perceptron
* CNN (Simulated via MLP)
* LSTM (Simulated via MLP)
* CNN-LSTM (Baseline Paper Model)

---

##  Key Findings

* Proposed model outperforms baseline on clean test data.
* Significant robustness improvement at **20 dB noise level**.
* Multi-scale wavelet features improve disturbance detection.
* Graph-based modeling captures electrical topology effects.
* Cross-attention enhances spatial-temporal fusion.

---

##  Applications

* Smart Grid Monitoring
* Real-Time Fault Detection
* Power System Stability Analysis
* Wide Area Monitoring Systems (WAMS)

---



