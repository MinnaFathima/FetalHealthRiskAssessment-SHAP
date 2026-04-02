# 🤰 Fetal Health Prediction using Explainable AI

An end-to-end machine learning system for fetal health classification using Cardiotocography (CTG) data, enhanced with SHAP-based explainability, real-time dashboard visualization, and automated clinical reporting.

---

## 🚀 Project Overview

This project predicts fetal health status — **Normal, Suspect, or Pathological** — using machine learning on CTG data.
Unlike traditional models, it focuses not only on prediction accuracy but also on **interpretability and clinical usability**.

---

## 🎯 Key Features

* 🔍 Explainable AI using SHAP (local + global explanations)
* 🤖 Multi-class classification (Normal / Suspect / Pathological)
* 📊 Interactive dashboard using Streamlit
* 📁 Batch prediction via CSV upload
* 📈 Risk visualization (bar charts, gauge meter)
* 📄 Automated hospital-style PDF report generation
* ⚖️ Handles class imbalance using SMOTE

---

## 🏗️ System Architecture

![System Architecture](images/architecture.png)

### 🔍 Workflow Summary

* CTG data is preprocessed (cleaning, scaling, balancing)
* Random Forest model predicts fetal health class and probabilities
* SHAP explains model decisions (feature contributions)
* Results are visualized in a Streamlit dashboard
* Final outputs include risk levels and PDF reports

---

## 📊 Dataset

* Source: Kaggle (Fetal Health Classification Dataset)
* Total samples: 2126
* Features: 21 CTG parameters
* Target classes:

  * 0 → Normal
  * 1 → Suspect
  * 2 → Pathological

---

## 🧠 Model Details

* Algorithm: Random Forest Classifier
* Handles non-linear relationships in CTG data
* Provides feature importance for interpretability
* Works well with tabular medical datasets

---

## 📈 Model Performance

* ✅ Test Accuracy: **91%**
* ✅ ROC-AUC Score: **0.98**
* ✅ Evaluated using Confusion Matrix, ROC Curve, Precision-Recall Curve

---

## ⚠️ Challenges Addressed

* Class imbalance handled using SMOTE
* Difficulty in interpreting black-box models
* Need for clinically usable AI systems

---

## 💡 Novelty of the Project

* Combines **prediction + explainability + clinical reporting**
* Provides **multi-level explainability (global + patient-level)**
* Designed for **real-world healthcare usability**
* Supports both **single and multi-patient analysis**

---

## 🖥️ Tech Stack

* Python
* Scikit-learn
* SHAP
* Streamlit
* Plotly
* Pandas, NumPy
* ReportLab (PDF generation)

---

## 📂 Project Structure

```
capstone/
│── fetal_dashboard.py
|── model_analysis.py
|── train_model.py
│── README.md

```

---

## ▶️ How to Run

```bash
streamlit run fetal_dashboard.py
```

---

## 📌 Future Improvements

* Integration with real-time hospital monitoring systems
* Deployment on cloud platforms
* Improving performance for “Suspect” class
* Integration with electronic health records (EHR)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It is not a substitute for professional medical diagnosis.

---

## 👩‍💻 Author

* Developed as part of M.Tech Software Engineering project
* Focused on AI in Healthcare & Explainable AI

---

## ⭐ Acknowledgment

* Kaggle for dataset
* Open-source ML and visualization libraries
