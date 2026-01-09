# 📊 Customer Churn Prediction & Retention System

An end-to-end analytics and machine learning project that predicts customer churn and provides actionable insights to support retention strategies.

---

## 🔍 Problem Statement
Customer churn leads to direct revenue loss for organizations. Identifying customers who are likely to churn in advance helps businesses take proactive retention actions.

---

## 💡 Solution
This project uses machine learning to predict customer churn based on historical customer data and deploys the model using an interactive Streamlit dashboard for real-time prediction and decision support.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Workflow
1. Data preprocessing and feature engineering
2. Machine learning model training (Logistic Regression & Random Forest)
3. Model evaluation using accuracy, precision, recall
4. Model deployment using Streamlit dashboard
5. Business recommendations based on churn probability

---

## 📈 Model Performance
- Algorithm: Random Forest Classifier
- Accuracy: ~80%
- Evaluation Metrics: Precision, Recall, F1-score

---

## 🖥️ Streamlit Dashboard
The dashboard allows users to:
- Enter customer details
- View churn probability
- Receive retention recommendations


## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Train the model
python train_model.py

5️⃣ Run the Streamlit app
python -m streamlit run app.py
