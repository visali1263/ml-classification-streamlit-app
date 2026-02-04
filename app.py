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

# -------------------------------------------------
# Display uploaded dataset
# -------------------------------------------------
st.subheader("📄 Uploaded Dataset Preview")
st.dataframe(data.head())

st.write("Dataset Shape:", data.shape)
st.write("Feature Columns:", X.columns.tolist())

# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------
st.subheader("📊 Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")

with col2:
    st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
    st.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")

with col3:
    st.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")
    st.metric("ROC-AUC", auc_score if auc_score == "N/A" else f"{auc_score:.4f}")

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
st.write(cm)

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
st.subheader("📑 Classification Report")
st.text(classification_report(y_true, y_pred))


