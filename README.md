# 🏦 Loan Approval Prediction App

A Machine Learning web application built with **Streamlit** to predict whether a loan application is likely to be approved or rejected.

This project demonstrates a full ML lifecycle:

- 📊 Data exploration dashboard  
- 🤖 Model training & validation with multiple algorithms  
- 🔁 Cross-validation & hyperparameter tuning  
- 📈 ROC curve & confusion matrix comparison  
- 🏆 Automatic best model selection  
- 💾 Model persistence with metadata tracking  
- 🔮 Real-time prediction interface  

---

# 📂 Dataset

The dataset used in this project is a **public dataset from Kaggle**.

- Source: Kaggle (Loan Approval / Loan Prediction dataset)
- Link: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Type: Structured tabular dataset
- Target variable: `loan_status`

This dataset is freely available for educational and research purposes.

The model was trained entirely on this free Kaggle dataset.

---

# 🧠 Machine Learning Pipeline

The app includes a complete ML validation framework:

## ✅ Models Compared

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

## 🔥 Features Implemented

- Train/Test Split
- Cross-validation
- GridSearchCV (hyperparameter tuning)
- ROC-AUC comparison
- Confusion matrix visualization
- Automatic best model selection
- Best model auto-save (`best_model.pkl`)
- Metadata tracking (`best_model_meta.json`)

The prediction page always loads and uses the best validated model.

---

# 📊 App Pages

## 1️⃣ Dashboard

- Dataset overview
- Feature distributions
- Loan status comparison
- Interactive histograms
## 2️⃣ Data Entry

- User-friendly form to add a new loan application
- Input validation for numeric and categorical fields
- Newly added entry is appended to the dataset
- The new entry can immediately be used for prediction

## 3️⃣  Prediction
- Loads the best validated model
- Displays the selected applicant information
- Shows prediction result (Approved / Rejected)
- Displays active model metadata (model name, ROC-AUC, training time, hyperparameters)

## 4️⃣ Model Validation

- Runs multiple ML models
- Logs training progress
- Displays ROC curves
- Displays confusion matrices
- Selects and saves best-performing model

---

# 🚀 How to Run the App Locally

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/tam201197/loan-approval-app.git
cd loan-approval-app
```

## 2️⃣ Create Virtual Environment (Recommended)

Using venv:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

OR using Conda:

```bash
conda create -n loan-app python=3.12
conda activate loan-app
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

# 🌍 Deployment

This app can be deployed for free using:

- Streamlit Community Cloud

Make sure your repository contains:

- `requirements.txt`
- `app.py`
- `best_model/best_model.pkl`
- `best_model/best_model_meta.json`

---

# 🏗 Project Structure

```
Loan-Approval-App/
│
├── app.py
├── utils.py
├── best_model
│   ├──best_model.pkl
│   ├── best_model_meta.json
├── requirements.txt
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Prediction.py
│   ├── 3_Model_Validation.py
```

---

# 🎯 Purpose of This Project

This project demonstrates:

- Practical Machine Learning engineering
- Model benchmarking & evaluation
- ML lifecycle management
- Deployment-ready architecture
- Reproducible environment setup

It is designed as a portfolio-ready ML application.

---

# 📌 Disclaimer

This application is for educational purposes only.  
Predictions should not be used for real financial decision-making.

---

# 👨‍💻 Author

Tam Truong  
MSc Computer Science  
Interested in AI / Machine Learning / ML Engineering

---

⭐ If you find this project useful, consider giving it a star on GitHub!

