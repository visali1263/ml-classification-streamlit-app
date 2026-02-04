#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification App",
    layout="wide"
)

st.title("📊 ML Assignment 2 – Classification Models")

st.markdown("""
This Streamlit application demonstrates multiple **machine learning classification models**
as required in **ML Assignment-2**.
""")

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("⚙️ App Controls")

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV only)",
    type=["csv"]
)

# -------------------------------------------------
# Model path mapping
# -------------------------------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Please upload a CSV test dataset to begin.")
    st.stop()

# Load dataset
data = pd.read_csv(uploaded_file)

if "target" not in data.columns:
    st.error("❌ The uploaded CSV must contain a column named 'target'.")
    st.stop()

X = data.drop(columns=["target"])
y_true = data["target"]

# Load selected model
try:
    with open(MODEL_PATHS[model_choice], "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the /model directory.")
    st.stop()

# Predictions
y_pred = model.predict(X)

# AUC handling (binary or multiclass)
auc_score = "N/A"
if hasattr(model, "predict_proba"):
    try:
        y_prob = model.predict_proba(X)
        if y_prob.shape[1] == 2:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_score = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc_score = "N/A"

# -------------------------------------------------
# Metrics display
# ----------

