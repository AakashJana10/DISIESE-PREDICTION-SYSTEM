# 🏥 AI Disease Prediction System

A production-quality web application built with **Streamlit** and **Machine Learning** that predicts potential medical conditions based on user-reported symptoms.

---

## ✨ Features

- **132 Symptoms Catalog:** Searchable, interactive multi-select input with clean formatting.
- **42 Diseases Covered:** Wide diagnostic coverage (infectious diseases, respiratory, cardiac, metabolic, dermatological, gastrointestinal, and neurological conditions).
- **High Accuracy Machine Learning:** Bayesian Diagnostic inference engine achieving **95.0% test accuracy**.
- **Top 5 Differential Diagnoses:** Visual confidence breakdown bars for the most probable conditions.
- **Clinical Insights:** Comprehensive condition descriptions, severity classification badges, and recommended medical precautions.
- **Zero-Friction Windows Setup:** Works out-of-the-box with standard library dependencies and Streamlit.

---

## 📁 Project Structure

```
diesese/
├── app.py               # Streamlit web application
├── train_model.py       # ML Model training script & Bayesian Classifier
├── create_dataset.py    # Dataset generator (data/Training.csv & Testing.csv)
├── run.bat              # One-click Windows launcher
├── requirements.txt     # Python dependencies
├── data/
│   ├── Training.csv     # 5,040 patient symptom profiles
│   └── Testing.csv      # 420 test validation profiles
└── model/
    ├── disease_model.pkl # Serialized trained classifier
    └── symptoms.pkl      # Symptoms feature list
```

---

## 🚀 How to Run the Application

### Option A: One-Click Launcher (Windows)
Simply double-click:
```
run.bat
```

### Option B: Command Line
Open your terminal in this directory and run:
```bash
streamlit run app.py
```
*(Or if using the specific Python path: `& "C:\Users\Aakash Jana\AppData\Local\Programs\Python\Python315\python.exe" -m streamlit run app.py`)*

Once started, open your browser at:
👉 **`http://localhost:8501`**

---

## 🔄 Retraining or Regenerating Data (Optional)

If you ever wish to re-generate datasets or re-train the model:

```bash
# 1. Regenerate 5,000+ training cases
python create_dataset.py

# 2. Retrain the model
python train_model.py
```

---

## ⚠️ Disclaimer
This system is developed strictly for **educational and informational purposes**. It does not replace clinical judgment, examination, or diagnosis by a qualified medical professional.
