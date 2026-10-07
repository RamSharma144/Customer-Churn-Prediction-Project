# Customer Churn Prediction

## Project Overview
This project uses machine learning to predict whether a customer is likely to leave (churn) a service.

## Objectives
- Clean and explore customer data
- Prepare numerical and categorical features
- Train a classification model
- Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix
- Generate a churn prediction for a new customer

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Machine Learning Model
A Logistic Regression classifier is used with:
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Median/most-frequent imputation for missing values
- Balanced class weights

## Dataset
The included `data/customer_churn.csv` is a compact demonstration dataset created for this project. For a larger portfolio version, it can be replaced with a public telecom/customer churn dataset while keeping the same pipeline.

## Project Structure
```text
customer-churn-prediction/
├── data/
│   └── customer_churn.csv
├── notebooks/
├── src/
│   └── train_model.py
├── README.md
├── requirements.txt
└── LICENSE
```

## How to Run
```bash
git clone https://github.com/RamSharma144/Customer-Churn-Prediction-Project.git
cd Customer-Churn-Prediction-Project
pip install -r requirements.txt
python src/train_model.py
```

## Key Learning Outcomes
- Exploratory data analysis
- Feature preprocessing
- Classification
- Model evaluation
- Business interpretation of churn predictions

## Disclaimer
This is an educational machine-learning project. Predictions are demonstrations and should not be used as the sole basis for real customer-retention decisions.
