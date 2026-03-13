#  Wavelet-GAT-MCA

### Noise-Robust Power System Event Classification using Graph Attention and Wavelet Features

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Signal Processing](https://img.shields.io/badge/Signal%20Processing-Wavelet-green)
![Graph ML](https://img.shields.io/badge/Graph%20Learning-GAT-red)

</p>

---

#  Project Overview

Modern smart grids rely on **Phasor Measurement Units (PMUs)** to monitor voltage and current signals across the network.
However, real-world PMU measurements often contain **noise and subtle disturbances**, making event detection a challenging machine learning task.

This project proposes a **hybrid AI architecture** combining:

* **Wavelet-based signal encoding**
* **Graph-based spatial modeling**
* **Attention-based feature fusion**

to perform **robust power system event classification under noisy conditions**.

The system is evaluated against multiple ML baselines and benchmark models to demonstrate improved **noise robustness and classification performance**.

---

#  Why This Project Matters

This project demonstrates a **complete real-world machine learning workflow**, including:

*  Time-series signal processing
*  Advanced feature engineering
*  Graph-based modeling
*  Attention mechanisms
*  Multi-model benchmarking
*  Robustness testing under noise
*  Visualization and reporting

These are **key skills required for Data Science / ML engineering roles**.

---

# 🧠 Model Architecture

The proposed **Wavelet-GAT-MCA** architecture combines signal processing and machine learning techniques.

```id="arch1"
PMU Time-Series Signals
        │
        ▼
Wavelet Transform (CWT)
+ Gramian Angular Field (GAF)
        │
        ▼
Multi-Channel Image Representation (64×64×3)
        │
        ▼
Graph Attention Aggregation
(IEEE-39 Bus Topology)
        │
        ▼
Temporal Statistical Feature Extraction
        │
        ▼
Cross-Attention Feature Fusion
        │
        ▼
Deep MLP Classifier
        │
        ▼
Power System Event Prediction
```

---

#  Event Classification Problem

The model classifies six different power system conditions.

| Label  | Event Type           |
| ------ | -------------------- |
| Stable | Normal Operation     |
| GT     | Generator Trip       |
| LD     | Load Disconnection   |
| AG     | Single Phase Fault   |
| AB     | Phase-to-Phase Fault |
| ABCG   | Three-Phase Fault    |

---

#  Dataset Generation

Since real PMU datasets are limited, this project simulates realistic signals using an **IEEE-39 Bus Power System Model**.

Dataset characteristics:

* **250 samples per event**
* **Multi-phase voltage signals**
* **39 bus network topology**
* **Noise injected during training and testing**

Training Noise:

```
25–35 dB
```

Testing Noise Levels:

```
40 dB
35 dB
30 dB
25 dB
20 dB
```

This allows realistic evaluation of **model robustness**.

---

#  Feature Engineering

### Wavelet Transform

Captures **multi-scale temporal patterns** in electrical signals.

### Gramian Angular Field

Transforms time-series into **image representations** preserving temporal correlations.

### Statistical Feature Extraction

Extracted features include:

* Mean
* Standard deviation
* Min / Max
* Median
* Percentiles
* Channel-wise statistics

---

# 🤖 Models Implemented

## Proposed Model

**Wavelet-GAT-MCA**

Key components:

* Wavelet encoding
* Graph-based feature aggregation
* Cross-attention feature fusion
* Deep neural classifier

---

## Baseline Model

CNN-LSTM style hybrid model.

---

## Benchmark Models

The project compares performance with:

* K-Nearest Neighbors
* Support Vector Machine
* Decision Tree
* Multi-Layer Perceptron
* CNN (simulated)
* LSTM (simulated)
* CNN-LSTM Hybrid

---

# 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# 📉 Noise Robustness Evaluation

A key contribution of this project is **testing model performance under varying noise levels**.

The proposed model maintains higher accuracy compared to baseline models even under **very high noise (20 dB)**.

---

#  Example Outputs

Running the project automatically generates:

```
model_comparison_table.csv
noise_resistance_table.csv
model_comparison.png
noise_resistance_comparison.png
confusion_matrix.png
```

These outputs provide:

* model performance comparison
* noise robustness evaluation
* visual analytics

---

#  Tech Stack

### Programming

Python

### Libraries

* NumPy
* Pandas
* SciPy
* Scikit-learn
* PyWavelets
* Matplotlib
* Seaborn

### Concepts Demonstrated

* Time-Series Analysis
* Signal Processing
* Feature Engineering
* Graph-based Machine Learning
* Attention Mechanisms
* Model Benchmarking
* Noise Robustness Testing

---

# 📂 Project Structure

```id="structure"
Wavelet-GAT-MCA/
│
├── complex.py
├── model_comparison_table.csv
├── noise_resistance_table.csv
├── model_comparison.png
├── noise_resistance_comparison.png
├── confusion_matrix.png
└── README.md
```

---

# ▶️ Running the Project

### Clone the repository

```bash id="clone"
git clone https://github.com/rajeev-333/Wavelet-Graph-Attention-with-Cross-Attention-Fusion-for-Noise-Resilient-Power-System-Event-Detection.git
cd Wavelet-GAT-MCA
```

### Install dependencies

```bash id="install"
pip install numpy pandas matplotlib seaborn scipy scikit-learn pywavelets
```

### Run the experiment

```bash id="run"
python complex.py
```

---

# 📌 Key Takeaways

This project demonstrates practical experience in:

* Designing **ML pipelines for time-series data**
* Handling **noisy real-world sensor signals**
* Implementing **graph-based feature aggregation**
* Building **robust classification systems**
* Comparing and evaluating **multiple ML models**

---

# 👨‍💻 Author

**Rajeev Gupta**

Interests:

* Machine Learning
* Data Science
* Time-Series Analytics
* AI for Power Systems

---




