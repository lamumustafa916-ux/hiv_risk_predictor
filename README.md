# 🔴 HIV Risk Prediction System
### Machine Learning Classification Project — Tanzania 2026

A machine learning web application that predicts HIV risk level (**High Risk / Low Risk**) based on behavioral, demographic, and healthcare factors relevant to the Tanzanian population.

---

## 📁 Project Structure

```
├── ml_project.ipynb        # Jupyter Notebook (full ML pipeline)
├── app.py                  # Streamlit web application
├── model.pkl               # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler for feature normalization
├── encoders.pkl            # Label encoders for categorical features
├── hiv_risk_dataset.csv    # Dataset (1,000 records, 18 columns)
├── project_report.docx     # Project report (2 pages)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🎯 Objective

Build a classification model that predicts whether an individual is at **High Risk** or **Low Risk** of HIV exposure based on 17 input features including:
- Behavioral factors (condom use, number of partners, IV drug use)
- Demographic factors (age, gender, education, region)
- Healthcare factors (STI history, HIV testing, healthcare access)

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Records  | 1,000 rows |
| Features | 17 input features |
| Target   | `hiv_risk` — High Risk / Low Risk |
| Balance  | 40% High Risk · 60% Low Risk |

---

## 🤖 Models Trained

| Model | Accuracy | ROC-AUC | CV Accuracy |
|-------|----------|---------|-------------|
| **Logistic Regression** ✅ | **84.00%** | **0.9167** | **~83.5%** |
| Decision Tree | 80.50% | 0.8611 | ~79.8% |

> **Logistic Regression** was selected as the best model.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/hiv-risk-predictor.git
cd hiv-risk-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push all files to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your repo
4. Set main file to `app.py` → click **Deploy**

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is NOT a substitute for professional HIV testing or medical advice. For actual HIV testing, please visit a certified healthcare facility.

---

## 👥 Group Members

| # | Name | Student ID | Contribution |
|---|------|------------|--------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## 📅 Timeline

| Milestone | Date |
|-----------|------|
| **Submission** | **16 February 2026** |
| **Presentation** | **18 February 2026** |
