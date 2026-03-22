# 5G Intrusion Detection System (IDS)

## 📌 Overview

This project builds a machine learning-based Intrusion Detection System (IDS) for 5G network traffic.
It processes a dataset, separates benign and attack traffic, and trains SVM models to detect intrusions.

---

## ⚙️ What the Code Does

* Splits the dataset into **benign and attack traffic**
* Groups attacks into **7 categories**
* Creates **training (80%) and testing (20%) datasets**
* Preprocesses and cleans the data
* Trains **SVM models (Linear & RBF kernels)**
* Evaluates performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Saves trained models and confusion matrices

---

## 🚀 How to Run

1. Install required libraries:

```
pip install pandas numpy scikit-learn matplotlib joblib
```

2. Place your dataset file:

```
Combined.csv
```

3. Run:

```
python main.py
```

---

## 📂 Output

* Processed datasets (Benign, Attacks, Categories)
* Training & evaluation splits
* Trained SVM models (`.pkl`)
* Confusion matrix plots

---

## 📊 Notes

* Dataset is not included due to size
* Models are trained per category for binary classification (benign vs attack)

---

## 👤 Author

Marios Clark
